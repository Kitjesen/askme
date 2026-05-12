# AskMe Video Lab

Small Remotion workspace for making videos from generated images, screenshots, captions, and timelines.

## Use with image2 / image generation

1. Put generated images in `public/assets/image2/`.
2. Edit `src/videoConfig.ts` and set each scene `image` to a path like `assets/image2/scene-01.png`.
3. Preview:

```bash
npm.cmd run dev
```

4. Render:

```bash
npm.cmd run render
```

The default composition also works without images by rendering polished placeholder scenes, so the pipeline can be tested before assets exist.
