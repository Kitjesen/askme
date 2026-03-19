"""Insert WAV audio files into PPT slides via PowerPoint COM automation."""

import os
import sys
import time

# Audio files to insert per slide (slide_number, list of (wav_file, y_inches))
# Slide 9: voice demo upper
# Slide 10: voice demo lower
AUDIO_MAP = {
    9: [
        ("01_greeting.wav", 1.55),
        ("02_confirm_task.wav", 3.35),
        ("04_safety_confirm.wav", 5.15),
    ],
    10: [
        ("05_patrol_report.wav", 1.55),
        ("03_estop.wav", 3.35),
        ("06_memory_demo.wav", 5.15),
    ],
}

X_INCHES = 10.85  # align with the ▶ play buttons
ICON_SIZE = 0.5   # inches


def main():
    import win32com.client

    base_dir = os.path.dirname(os.path.abspath(__file__))
    speaker_dir = os.path.join(base_dir, "voice_samples", "speaker_10")
    pptx_path = os.path.join(base_dir, "NOVA_Dog_Askme_v4_Report_v4.pptx")
    out_path = os.path.join(base_dir, "NOVA_Dog_Askme_v4_Report_v5.pptx")

    if not os.path.exists(pptx_path):
        print(f"ERROR: {pptx_path} not found")
        sys.exit(1)

    print("Starting PowerPoint...")
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True  # must be visible for AddMediaObject2

    try:
        print(f"Opening {pptx_path}...")
        prs = ppt.Presentations.Open(pptx_path)
        time.sleep(1)

        total = 0
        for slide_num, audio_list in AUDIO_MAP.items():
            if slide_num > prs.Slides.Count:
                print(f"  WARNING: Slide {slide_num} doesn't exist (only {prs.Slides.Count} slides)")
                continue

            slide = prs.Slides(slide_num)
            print(f"\nSlide {slide_num}:")

            for wav_name, y_inches in audio_list:
                wav_path = os.path.join(speaker_dir, wav_name)
                if not os.path.exists(wav_path):
                    print(f"  SKIP: {wav_path} not found")
                    continue

                # Convert inches to points (1 inch = 72 points)
                left = X_INCHES * 72
                top = y_inches * 72
                width = ICON_SIZE * 72
                height = ICON_SIZE * 72

                try:
                    shape = slide.Shapes.AddMediaObject2(
                        wav_path,
                        False,   # LinkToFile = False (embed)
                        True,    # SaveWithDocument = True
                        left, top, width, height,
                    )
                    # Set to play on click (not auto)
                    # AnimationSettings: ppAnimateOnClick = 1
                    shape.AnimationSettings.PlaySettings.PlayOnEntry = False
                    shape.AnimationSettings.PlaySettings.HideWhileNotPlaying = False

                    print(f"  Inserted: {wav_name} at ({X_INCHES:.1f}\", {y_inches:.1f}\")")
                    total += 1
                except Exception as exc:
                    print(f"  ERROR inserting {wav_name}: {exc}")

        print(f"\nSaving to {out_path}...")
        prs.SaveAs(out_path)
        prs.Close()
        print(f"\nDone! {total} audio files embedded.")
        print(f"Output: {out_path}")

    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        # Don't quit PowerPoint in case user wants to review
        pass


if __name__ == "__main__":
    main()
