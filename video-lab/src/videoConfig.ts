import type {Image2StoryboardProps} from './Image2Storyboard';

export const videoConfig = {
  fps: 30,
  width: 1920,
  height: 1080,
  durationInFrames: 30 * 18,
  props: {
    title: 'AskMe Video Lab',
    subtitle: 'Image2-ready motion storyboard',
    footer: 'Drop generated frames into public/assets/image2',
    accent: '#31d0aa',
    scenes: [
      {
        label: 'Concept',
        headline: 'Prompt becomes a shot plan',
        detail: 'Script, timing, and visual direction are kept editable in code.',
        image: '',
      },
      {
        label: 'Image2',
        headline: 'Generated frames become scenes',
        detail: 'Use image2 or image_gen outputs as backgrounds, inserts, or hero art.',
        image: '',
      },
      {
        label: 'Render',
        headline: 'Remotion composes the final MP4',
        detail: 'Captions, transitions, camera motion, audio, and overlays stay deterministic.',
        image: '',
      },
    ],
  } satisfies Image2StoryboardProps,
};
