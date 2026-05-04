#include <libusb-1.0/libusb.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MCP01_VID 0x17ef
#define MCP01_PID 0xa03b

#define IFACE_CONTROL 1
#define IFACE_PLAYBACK 2
#define IFACE_CAPTURE 3

#define EP_PLAYBACK 0x02
#define EP_CAPTURE 0x81

#define PLAY_RATE 48000
#define PLAY_CHANNELS 2
#define PLAY_PACKET_FRAMES 48
#define PLAY_PACKET_SIZE (PLAY_PACKET_FRAMES * PLAY_CHANNELS * 2)
#define PLAY_PACKETS_PER_TRANSFER 8
#define PLAY_TRANSFERS 8
#define STREAM_RING_SIZE (PLAY_RATE * PLAY_CHANNELS * 2)
#define STREAM_EOF_TAIL_PACKETS 80

#define CAPTURE_PACKET_SIZE 32
#define CAPTURE_PACKETS_PER_TRANSFER 16
#define CAPTURE_TRANSFERS 8

struct play_state {
    int next_packet;
    int total_packets;
    int active;
    int errors;
    int amp;
    int stdin_mode;
    const unsigned char *pcm;
    int pcm_len;
    int pcm_pos;
    double freq;
    double phase;
};

struct play_stream_state {
    int active;
    int errors;
    int eof;
    int submitted_packets;
    int stdin_bytes;
    int silence_packets;
    int tail_packets_remaining;
    unsigned char ring[STREAM_RING_SIZE];
    int ring_start;
    int ring_len;
};

struct capture_state {
    int packet_budget;
    int continuous;
    int submitted_packets;
    int completed_packets;
    int active;
    int errors;
    int raw_errors;
    long long sumsq;
    int samples;
    int max_abs;
    FILE *raw_out;
};

static libusb_device_handle *open_mcp01(libusb_context *ctx) {
    libusb_device_handle *handle = libusb_open_device_with_vid_pid(ctx, MCP01_VID, MCP01_PID);
    if (handle == NULL) {
        fprintf(stderr, "MCP01 %04x:%04x not found or not openable\n", MCP01_VID, MCP01_PID);
        return NULL;
    }
    libusb_set_auto_detach_kernel_driver(handle, 1);
    return handle;
}

static void close_claimed(libusb_device_handle *handle, int streaming_iface) {
    if (handle == NULL) return;
    if (streaming_iface >= 0) {
        libusb_set_interface_alt_setting(handle, streaming_iface, 0);
        libusb_release_interface(handle, streaming_iface);
    }
    libusb_release_interface(handle, IFACE_CONTROL);
    libusb_close(handle);
}

static int claim_interfaces(libusb_device_handle *handle, int streaming_iface) {
    int r = libusb_claim_interface(handle, IFACE_CONTROL);
    if (r < 0) {
        fprintf(stderr, "claim control interface warning: %s\n", libusb_error_name(r));
    }
    r = libusb_claim_interface(handle, streaming_iface);
    if (r < 0) {
        fprintf(stderr, "claim streaming interface %d: %s\n", streaming_iface, libusb_error_name(r));
        libusb_release_interface(handle, IFACE_CONTROL);
        return r;
    }
    r = libusb_set_interface_alt_setting(handle, streaming_iface, 1);
    if (r < 0) {
        fprintf(stderr, "set interface %d alt=1: %s\n", streaming_iface, libusb_error_name(r));
        libusb_release_interface(handle, streaming_iface);
        libusb_release_interface(handle, IFACE_CONTROL);
        return r;
    }
    return 0;
}

static void configure_playback(libusb_device_handle *handle) {
    unsigned char freq_48k[3] = {0x80, 0xbb, 0x00};
    unsigned char mute = 0;
    unsigned char volume[2] = {0xf0, 0xff};
    int r;

    r = libusb_control_transfer(handle, 0x22, 0x01, 0x0100, EP_PLAYBACK,
                                freq_48k, sizeof(freq_48k), 1000);
    fprintf(stderr, "set endpoint sample rate result=%d\n", r);

    r = libusb_control_transfer(handle, 0x21, 0x01, 0x0100,
                                (9 << 8) | IFACE_CONTROL, &mute, sizeof(mute), 1000);
    fprintf(stderr, "set speaker mute=0 result=%d\n", r);

    r = libusb_control_transfer(handle, 0xa1, 0x83, 0x0200,
                                (9 << 8) | IFACE_CONTROL, volume, sizeof(volume), 1000);
    if (r == (int)sizeof(volume)) {
        fprintf(stderr, "speaker max volume raw=%02x%02x\n", volume[0], volume[1]);
    } else {
        fprintf(stderr, "get speaker max volume result=%d; using cached MCP01 max\n", r);
        volume[0] = 0xf0;
        volume[1] = 0xff;
    }

    r = libusb_control_transfer(handle, 0x21, 0x01, 0x0200,
                                (9 << 8) | IFACE_CONTROL, volume, sizeof(volume), 1000);
    fprintf(stderr, "set speaker volume=max result=%d\n", r);
}

static void fill_play_tone(struct play_state *state, unsigned char *buffer) {
    const double step = 2.0 * M_PI * state->freq / PLAY_RATE;
    int16_t *samples = (int16_t *)buffer;
    for (int p = 0; p < PLAY_PACKETS_PER_TRANSFER; ++p) {
        for (int f = 0; f < PLAY_PACKET_FRAMES; ++f) {
            int16_t sample = (int16_t)(sin(state->phase) * (double)state->amp);
            samples[0] = sample;
            samples[1] = sample;
            samples += 2;
            state->phase += step;
            if (state->phase >= 2.0 * M_PI) state->phase -= 2.0 * M_PI;
        }
    }
}

static void fill_play_pcm(struct play_state *state, unsigned char *buffer, int packet_count) {
    int byte_count = packet_count * PLAY_PACKET_SIZE;
    int remaining = state->pcm_len - state->pcm_pos;
    int copy_len = remaining < byte_count ? remaining : byte_count;

    memset(buffer, 0, PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE);
    if (copy_len > 0) {
        memcpy(buffer, state->pcm + state->pcm_pos, (size_t)copy_len);
        state->pcm_pos += copy_len;
    }
}

static int submit_play_transfer(struct libusb_transfer *transfer, struct play_state *state);

static void play_callback(struct libusb_transfer *transfer) {
    struct play_state *state = (struct play_state *)transfer->user_data;
    if (transfer->status != LIBUSB_TRANSFER_COMPLETED) {
        fprintf(stderr, "play transfer status=%d actual=%d\n",
                transfer->status, transfer->actual_length);
        state->errors++;
    }
    if (state->next_packet < state->total_packets) {
        if (submit_play_transfer(transfer, state) == 0) return;
        state->errors++;
    }
    state->active--;
}

static int submit_play_transfer(struct libusb_transfer *transfer, struct play_state *state) {
    int remaining = state->total_packets - state->next_packet;
    int packets = PLAY_PACKETS_PER_TRANSFER;
    if (remaining <= 0) return -1;
    if (remaining < packets) packets = remaining;

    if (state->stdin_mode) {
        fill_play_pcm(state, transfer->buffer, packets);
    } else {
        memset(transfer->buffer, 0, PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE);
        fill_play_tone(state, transfer->buffer);
    }
    transfer->length = packets * PLAY_PACKET_SIZE;
    transfer->num_iso_packets = packets;
    for (int i = 0; i < packets; ++i) {
        transfer->iso_packet_desc[i].length = PLAY_PACKET_SIZE;
        transfer->iso_packet_desc[i].actual_length = 0;
        transfer->iso_packet_desc[i].status = 0;
    }
    state->next_packet += packets;
    return libusb_submit_transfer(transfer);
}

static int run_play(int ms, int amp, double freq) {
    libusb_context *ctx = NULL;
    libusb_device_handle *handle = NULL;
    struct play_state state;
    struct libusb_transfer *transfers[PLAY_TRANSFERS];
    int result = 0;

    memset(&state, 0, sizeof(state));
    memset(transfers, 0, sizeof(transfers));
    state.total_packets = ms;
    state.amp = amp;
    state.freq = freq;

    int r = libusb_init(&ctx);
    if (r < 0) {
        fprintf(stderr, "libusb_init: %s\n", libusb_error_name(r));
        return 1;
    }
    handle = open_mcp01(ctx);
    if (handle == NULL) {
        libusb_exit(ctx);
        return 2;
    }
    r = claim_interfaces(handle, IFACE_PLAYBACK);
    if (r < 0) {
        libusb_close(handle);
        libusb_exit(ctx);
        return 3;
    }
    configure_playback(handle);

    for (int i = 0; i < PLAY_TRANSFERS; ++i) {
        unsigned char *buffer = calloc(1, PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE);
        transfers[i] = libusb_alloc_transfer(PLAY_PACKETS_PER_TRANSFER);
        if (buffer == NULL || transfers[i] == NULL) {
            fprintf(stderr, "play allocation failed\n");
            free(buffer);
            result = 4;
            goto cleanup;
        }
        libusb_fill_iso_transfer(transfers[i], handle, EP_PLAYBACK, buffer,
                                 PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE,
                                 PLAY_PACKETS_PER_TRANSFER, play_callback, &state, 1000);
        if (state.next_packet >= state.total_packets) break;
        r = submit_play_transfer(transfers[i], &state);
        if (r == 0) {
            state.active++;
        } else {
            fprintf(stderr, "initial play submit failed: %s\n", libusb_error_name(r));
            state.errors++;
        }
    }

    fprintf(stderr, "streaming %d ms tone to MCP01 USB endpoint 0x%02x...\n", ms, EP_PLAYBACK);
    while (state.active > 0) {
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000;
        r = libusb_handle_events_timeout(ctx, &tv);
        if (r < 0) {
            fprintf(stderr, "play events: %s\n", libusb_error_name(r));
            state.errors++;
            break;
        }
    }

cleanup:
    for (int i = 0; i < PLAY_TRANSFERS; ++i) {
        if (transfers[i] != NULL) {
            free(transfers[i]->buffer);
            libusb_free_transfer(transfers[i]);
        }
    }
    close_claimed(handle, IFACE_PLAYBACK);
    libusb_exit(ctx);
    printf("play_done sent_packets=%d errors=%d\n", state.next_packet, state.errors);
    if (result != 0) return result;
    return state.errors ? 5 : 0;
}

static int run_play_pcm(const unsigned char *pcm, int pcm_len) {
    libusb_context *ctx = NULL;
    libusb_device_handle *handle = NULL;
    struct play_state state;
    struct libusb_transfer *transfers[PLAY_TRANSFERS];
    int result = 0;

    memset(&state, 0, sizeof(state));
    memset(transfers, 0, sizeof(transfers));
    state.stdin_mode = 1;
    state.pcm = pcm;
    state.pcm_len = pcm_len;
    state.total_packets = (pcm_len + PLAY_PACKET_SIZE - 1) / PLAY_PACKET_SIZE;

    if (state.total_packets <= 0) {
        printf("play_done sent_packets=0 errors=0\n");
        return 0;
    }

    int r = libusb_init(&ctx);
    if (r < 0) {
        fprintf(stderr, "libusb_init: %s\n", libusb_error_name(r));
        return 1;
    }
    handle = open_mcp01(ctx);
    if (handle == NULL) {
        libusb_exit(ctx);
        return 2;
    }
    r = claim_interfaces(handle, IFACE_PLAYBACK);
    if (r < 0) {
        libusb_close(handle);
        libusb_exit(ctx);
        return 3;
    }
    configure_playback(handle);

    for (int i = 0; i < PLAY_TRANSFERS; ++i) {
        unsigned char *buffer = calloc(1, PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE);
        transfers[i] = libusb_alloc_transfer(PLAY_PACKETS_PER_TRANSFER);
        if (buffer == NULL || transfers[i] == NULL) {
            fprintf(stderr, "play allocation failed\n");
            free(buffer);
            result = 4;
            goto cleanup;
        }
        libusb_fill_iso_transfer(transfers[i], handle, EP_PLAYBACK, buffer,
                                 PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE,
                                 PLAY_PACKETS_PER_TRANSFER, play_callback, &state, 1000);
        if (state.next_packet >= state.total_packets) break;
        r = submit_play_transfer(transfers[i], &state);
        if (r == 0) {
            state.active++;
        } else {
            fprintf(stderr, "initial play submit failed: %s\n", libusb_error_name(r));
            state.errors++;
        }
    }

    fprintf(stderr, "streaming %d bytes PCM to MCP01 USB endpoint 0x%02x...\n",
            pcm_len, EP_PLAYBACK);
    while (state.active > 0) {
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000;
        r = libusb_handle_events_timeout(ctx, &tv);
        if (r < 0) {
            fprintf(stderr, "play events: %s\n", libusb_error_name(r));
            state.errors++;
            break;
        }
    }

cleanup:
    for (int i = 0; i < PLAY_TRANSFERS; ++i) {
        if (transfers[i] != NULL) {
            free(transfers[i]->buffer);
            libusb_free_transfer(transfers[i]);
        }
    }
    close_claimed(handle, IFACE_PLAYBACK);
    libusb_exit(ctx);
    printf("play_done sent_packets=%d bytes=%d errors=%d\n",
           state.next_packet, state.pcm_pos, state.errors);
    if (result != 0) return result;
    return state.errors ? 5 : 0;
}

static void stream_ring_append(struct play_stream_state *state,
                               const unsigned char *data, int len) {
    while (len > 0 && state->ring_len < STREAM_RING_SIZE) {
        int write_pos = (state->ring_start + state->ring_len) % STREAM_RING_SIZE;
        int space_to_end = STREAM_RING_SIZE - write_pos;
        int space_total = STREAM_RING_SIZE - state->ring_len;
        int chunk = len;
        if (chunk > space_to_end) chunk = space_to_end;
        if (chunk > space_total) chunk = space_total;
        memcpy(state->ring + write_pos, data, (size_t)chunk);
        state->ring_len += chunk;
        data += chunk;
        len -= chunk;
    }
}

static int stream_ring_pop(struct play_stream_state *state,
                           unsigned char *dest, int len) {
    int copied = 0;
    while (len > 0 && state->ring_len > 0) {
        int chunk = state->ring_len;
        int to_end = STREAM_RING_SIZE - state->ring_start;
        if (chunk > to_end) chunk = to_end;
        if (chunk > len) chunk = len;
        memcpy(dest + copied, state->ring + state->ring_start, (size_t)chunk);
        state->ring_start = (state->ring_start + chunk) % STREAM_RING_SIZE;
        state->ring_len -= chunk;
        copied += chunk;
        len -= chunk;
    }
    return copied;
}

static void read_stream_stdin(struct play_stream_state *state) {
    unsigned char buf[8192];

    while (!state->eof && state->ring_len < STREAM_RING_SIZE) {
        int available = STREAM_RING_SIZE - state->ring_len;
        int request = (int)sizeof(buf);
        if (request > available) request = available;
        ssize_t n = read(STDIN_FILENO, buf, (size_t)request);
        if (n > 0) {
            stream_ring_append(state, buf, (int)n);
            state->stdin_bytes += (int)n;
            continue;
        }
        if (n == 0) {
            state->eof = 1;
            return;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        fprintf(stderr, "stdin stream read failed: %s\n", strerror(errno));
        state->errors++;
        state->eof = 1;
        return;
    }
}

static int set_stdin_nonblocking(void) {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}

static int submit_play_stream_transfer(struct libusb_transfer *transfer,
                                       struct play_stream_state *state);

static void play_stream_callback(struct libusb_transfer *transfer) {
    struct play_stream_state *state = (struct play_stream_state *)transfer->user_data;
    if (transfer->status != LIBUSB_TRANSFER_COMPLETED) {
        fprintf(stderr, "play stream transfer status=%d actual=%d\n",
                transfer->status, transfer->actual_length);
        state->errors++;
    }
    if (submit_play_stream_transfer(transfer, state) == 0) return;
    state->active--;
}

static int submit_play_stream_transfer(struct libusb_transfer *transfer,
                                       struct play_stream_state *state) {
    read_stream_stdin(state);
    if (state->eof && state->ring_len == 0 && state->tail_packets_remaining <= 0) {
        return -1;
    }

    int packets = PLAY_PACKETS_PER_TRANSFER;
    int byte_count = packets * PLAY_PACKET_SIZE;
    memset(transfer->buffer, 0, PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE);
    int copied = stream_ring_pop(state, transfer->buffer, byte_count);
    if (copied < byte_count) {
        int silent_bytes = byte_count - copied;
        state->silence_packets += (silent_bytes + PLAY_PACKET_SIZE - 1) / PLAY_PACKET_SIZE;
        if (state->eof && state->ring_len == 0) {
            state->tail_packets_remaining -= packets;
            if (state->tail_packets_remaining < 0) state->tail_packets_remaining = 0;
        }
    }

    transfer->length = byte_count;
    transfer->num_iso_packets = packets;
    for (int i = 0; i < packets; ++i) {
        transfer->iso_packet_desc[i].length = PLAY_PACKET_SIZE;
        transfer->iso_packet_desc[i].actual_length = 0;
        transfer->iso_packet_desc[i].status = 0;
    }
    state->submitted_packets += packets;
    return libusb_submit_transfer(transfer);
}

static int run_play_stdin_stream(void) {
    libusb_context *ctx = NULL;
    libusb_device_handle *handle = NULL;
    struct play_stream_state state;
    struct libusb_transfer *transfers[PLAY_TRANSFERS];
    int result = 0;

    memset(&state, 0, sizeof(state));
    memset(transfers, 0, sizeof(transfers));
    state.tail_packets_remaining = STREAM_EOF_TAIL_PACKETS;

    if (set_stdin_nonblocking() < 0) {
        fprintf(stderr, "stdin nonblocking setup failed: %s\n", strerror(errno));
        return 1;
    }

    int r = libusb_init(&ctx);
    if (r < 0) {
        fprintf(stderr, "libusb_init: %s\n", libusb_error_name(r));
        return 1;
    }
    handle = open_mcp01(ctx);
    if (handle == NULL) {
        libusb_exit(ctx);
        return 2;
    }
    r = claim_interfaces(handle, IFACE_PLAYBACK);
    if (r < 0) {
        libusb_close(handle);
        libusb_exit(ctx);
        return 3;
    }
    configure_playback(handle);

    for (int i = 0; i < PLAY_TRANSFERS; ++i) {
        unsigned char *buffer = calloc(1, PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE);
        transfers[i] = libusb_alloc_transfer(PLAY_PACKETS_PER_TRANSFER);
        if (buffer == NULL || transfers[i] == NULL) {
            fprintf(stderr, "play stream allocation failed\n");
            free(buffer);
            result = 4;
            goto cleanup;
        }
        libusb_fill_iso_transfer(transfers[i], handle, EP_PLAYBACK, buffer,
                                 PLAY_PACKETS_PER_TRANSFER * PLAY_PACKET_SIZE,
                                 PLAY_PACKETS_PER_TRANSFER, play_stream_callback, &state, 1000);
        r = submit_play_stream_transfer(transfers[i], &state);
        if (r == 0) {
            state.active++;
        } else {
            fprintf(stderr, "initial play stream submit failed: %s\n", libusb_error_name(r));
            state.errors++;
        }
    }

    fprintf(stderr, "stdin stream ready for MCP01 USB endpoint 0x%02x...\n", EP_PLAYBACK);
    while (state.active > 0) {
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000;
        r = libusb_handle_events_timeout(ctx, &tv);
        if (r < 0) {
            fprintf(stderr, "play stream events: %s\n", libusb_error_name(r));
            state.errors++;
            break;
        }
    }

cleanup:
    for (int i = 0; i < PLAY_TRANSFERS; ++i) {
        if (transfers[i] != NULL) {
            free(transfers[i]->buffer);
            libusb_free_transfer(transfers[i]);
        }
    }
    close_claimed(handle, IFACE_PLAYBACK);
    libusb_exit(ctx);
    printf("stream_done submitted_packets=%d stdin_bytes=%d silence_packets=%d errors=%d\n",
           state.submitted_packets, state.stdin_bytes, state.silence_packets, state.errors);
    if (result != 0) return result;
    return state.errors ? 5 : 0;
}

static int submit_capture_transfer(struct libusb_transfer *transfer, struct capture_state *state);

static void capture_callback(struct libusb_transfer *transfer) {
    struct capture_state *state = (struct capture_state *)transfer->user_data;
    if (transfer->status != LIBUSB_TRANSFER_COMPLETED) {
        fprintf(stderr, "capture transfer status=%d actual=%d\n",
                transfer->status, transfer->actual_length);
        state->errors++;
    } else {
        for (int p = 0; p < transfer->num_iso_packets; ++p) {
            struct libusb_iso_packet_descriptor *desc = &transfer->iso_packet_desc[p];
            if (desc->status != LIBUSB_TRANSFER_COMPLETED) {
                state->errors++;
                continue;
            }
            unsigned char *data = libusb_get_iso_packet_buffer_simple(transfer, p);
            if (state->raw_out != NULL && desc->actual_length > 0) {
                size_t written = fwrite(data, 1, (size_t)desc->actual_length, state->raw_out);
                if (written != (size_t)desc->actual_length) {
                    state->errors++;
                    state->raw_errors++;
                    state->continuous = 0;
                    state->packet_budget = 0;
                    continue;
                }
            }
            int16_t *samples = (int16_t *)data;
            int count = desc->actual_length / 2;
            for (int i = 0; i < count; ++i) {
                int value = samples[i];
                int abs_value = value < 0 ? -value : value;
                if (abs_value > state->max_abs) state->max_abs = abs_value;
                state->sumsq += (long long)value * (long long)value;
                state->samples++;
            }
            state->completed_packets++;
        }
    }

    if (state->continuous || state->packet_budget > 0) {
        if (submit_capture_transfer(transfer, state) == 0) return;
        state->errors++;
    }
    state->active--;
}

static int submit_capture_transfer(struct libusb_transfer *transfer, struct capture_state *state) {
    int packets = CAPTURE_PACKETS_PER_TRANSFER;
    if (!state->continuous) {
        if (state->packet_budget <= 0) return -1;
        if (state->packet_budget < packets) packets = state->packet_budget;
        state->packet_budget -= packets;
    }

    memset(transfer->buffer, 0, CAPTURE_PACKETS_PER_TRANSFER * CAPTURE_PACKET_SIZE);
    transfer->length = packets * CAPTURE_PACKET_SIZE;
    transfer->num_iso_packets = packets;
    for (int i = 0; i < packets; ++i) {
        transfer->iso_packet_desc[i].length = CAPTURE_PACKET_SIZE;
        transfer->iso_packet_desc[i].actual_length = 0;
        transfer->iso_packet_desc[i].status = 0;
    }
    state->submitted_packets += packets;
    return libusb_submit_transfer(transfer);
}

static int run_capture(int ms, FILE *raw_out) {
    libusb_context *ctx = NULL;
    libusb_device_handle *handle = NULL;
    struct capture_state state;
    struct libusb_transfer *transfers[CAPTURE_TRANSFERS];
    int result = 0;

    memset(&state, 0, sizeof(state));
    memset(transfers, 0, sizeof(transfers));
    state.packet_budget = ms;
    state.continuous = (ms <= 0 && raw_out != NULL);
    state.raw_out = raw_out;

    if (raw_out != NULL) {
        setvbuf(raw_out, NULL, _IONBF, 0);
    }

    int r = libusb_init(&ctx);
    if (r < 0) {
        fprintf(stderr, "libusb_init: %s\n", libusb_error_name(r));
        return 1;
    }
    handle = open_mcp01(ctx);
    if (handle == NULL) {
        libusb_exit(ctx);
        return 2;
    }
    r = claim_interfaces(handle, IFACE_CAPTURE);
    if (r < 0) {
        libusb_close(handle);
        libusb_exit(ctx);
        return 3;
    }

    for (int i = 0; i < CAPTURE_TRANSFERS; ++i) {
        unsigned char *buffer = calloc(1, CAPTURE_PACKETS_PER_TRANSFER * CAPTURE_PACKET_SIZE);
        transfers[i] = libusb_alloc_transfer(CAPTURE_PACKETS_PER_TRANSFER);
        if (buffer == NULL || transfers[i] == NULL) {
            fprintf(stderr, "capture allocation failed\n");
            free(buffer);
            result = 4;
            goto cleanup;
        }
        libusb_fill_iso_transfer(transfers[i], handle, EP_CAPTURE, buffer,
                                 CAPTURE_PACKETS_PER_TRANSFER * CAPTURE_PACKET_SIZE,
                                 CAPTURE_PACKETS_PER_TRANSFER, capture_callback, &state, 1000);
        if (!state.continuous && state.packet_budget <= 0) break;
        r = submit_capture_transfer(transfers[i], &state);
        if (r == 0) {
            state.active++;
        } else {
            fprintf(stderr, "initial capture submit failed: %s\n", libusb_error_name(r));
            state.errors++;
        }
    }

    while (state.active > 0) {
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000;
        r = libusb_handle_events_timeout(ctx, &tv);
        if (r < 0) {
            fprintf(stderr, "capture events: %s\n", libusb_error_name(r));
            state.errors++;
            break;
        }
    }

cleanup:
    for (int i = 0; i < CAPTURE_TRANSFERS; ++i) {
        if (transfers[i] != NULL) {
            free(transfers[i]->buffer);
            libusb_free_transfer(transfers[i]);
        }
    }
    close_claimed(handle, IFACE_CAPTURE);
    libusb_exit(ctx);

    double rms = 0.0;
    if (state.samples > 0) {
        rms = sqrt((double)state.sumsq / (double)state.samples);
    }
    FILE *summary = raw_out != NULL ? stderr : stdout;
    fprintf(summary,
            "capture_done submitted_packets=%d completed_packets=%d samples=%d rms=%.2f max_abs=%d errors=%d raw_errors=%d\n",
            state.submitted_packets, state.completed_packets, state.samples,
            rms, state.max_abs, state.errors, state.raw_errors);

    if (result != 0) return result;
    return state.errors ? 5 : 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--play-ms N] [--capture-ms N] [--freq HZ] [--amp N] [--stdin-play] [--stdin-stream] [--capture-stdout]\n"
            "Default: play 3000ms tone and capture 3000ms from Lenovo MCP01 (17ef:a03b).\n"
            "--stdin-play reads 48kHz stereo S16_LE PCM from stdin and streams it to USB.\n"
            "--stdin-stream opens USB immediately, keeps streaming silence, and consumes stdin incrementally.\n",
            argv0);
}

static int read_stdin_all(unsigned char **out, int *out_len) {
    size_t cap = 65536;
    size_t len = 0;
    unsigned char *buf = malloc(cap);
    if (buf == NULL) {
        fprintf(stderr, "stdin allocation failed\n");
        return 1;
    }

    while (1) {
        if (len == cap) {
            size_t next_cap = cap * 2;
            unsigned char *next = realloc(buf, next_cap);
            if (next == NULL) {
                free(buf);
                fprintf(stderr, "stdin allocation failed\n");
                return 1;
            }
            buf = next;
            cap = next_cap;
        }

        size_t n = fread(buf + len, 1, cap - len, stdin);
        len += n;
        if (n == 0) {
            if (ferror(stdin)) {
                free(buf);
                fprintf(stderr, "stdin read failed\n");
                return 1;
            }
            break;
        }
    }

    if (len > 0x7fffffff) {
        free(buf);
        fprintf(stderr, "stdin PCM too large\n");
        return 1;
    }

    *out = buf;
    *out_len = (int)len;
    return 0;
}

int main(int argc, char **argv) {
    int play_ms = 3000;
    int capture_ms = 3000;
    int amp = 9000;
    double freq = 1000.0;
    int stdin_play = 0;
    int stdin_stream = 0;
    int capture_stdout = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--play-ms") == 0 && i + 1 < argc) {
            play_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--capture-ms") == 0 && i + 1 < argc) {
            capture_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--freq") == 0 && i + 1 < argc) {
            freq = atof(argv[++i]);
        } else if (strcmp(argv[i], "--amp") == 0 && i + 1 < argc) {
            amp = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stdin-play") == 0) {
            stdin_play = 1;
            play_ms = 0;
        } else if (strcmp(argv[i], "--stdin-stream") == 0) {
            stdin_stream = 1;
            play_ms = 0;
        } else if (strcmp(argv[i], "--capture-stdout") == 0) {
            capture_stdout = 1;
            play_ms = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 64;
        }
    }

    if (play_ms < 0 || capture_ms < 0 || amp < 0 || amp > 32767 || freq <= 0.0) {
        usage(argv[0]);
        return 64;
    }
    if (capture_stdout && (stdin_play || stdin_stream)) {
        fprintf(stderr, "--capture-stdout cannot be combined with playback\n");
        usage(argv[0]);
        return 64;
    }
    if (stdin_play && stdin_stream) {
        fprintf(stderr, "--stdin-play and --stdin-stream are mutually exclusive\n");
        usage(argv[0]);
        return 64;
    }

    int rc = 0;
    if (stdin_stream) {
        int play_rc = run_play_stdin_stream();
        if (play_rc != 0) rc = play_rc;
    } else if (stdin_play) {
        unsigned char *pcm = NULL;
        int pcm_len = 0;
        if (read_stdin_all(&pcm, &pcm_len) != 0) {
            return 1;
        }
        int play_rc = run_play_pcm(pcm, pcm_len);
        free(pcm);
        if (play_rc != 0) rc = play_rc;
    } else if (play_ms > 0) {
        int play_rc = run_play(play_ms, amp, freq);
        if (play_rc != 0) rc = play_rc;
    }
    if (capture_ms > 0 || capture_stdout) {
        int capture_rc = run_capture(capture_ms, capture_stdout ? stdout : NULL);
        if (capture_rc != 0 && rc == 0) rc = capture_rc;
    }
    return rc;
}
