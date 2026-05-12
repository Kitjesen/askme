import React from 'react';
import {
  AbsoluteFill,
  Img,
  interpolate,
  Sequence,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

export type StoryScene = {
  label: string;
  headline: string;
  detail: string;
  image?: string;
};

export type Image2StoryboardProps = {
  title: string;
  subtitle: string;
  footer: string;
  accent: string;
  scenes: StoryScene[];
};

const fontStack =
  'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

export const Image2Storyboard = ({
  title,
  subtitle,
  footer,
  accent,
  scenes,
}: Image2StoryboardProps) => {
  const {fps, durationInFrames} = useVideoConfig();
  const frame = useCurrentFrame();
  const introOpacity = interpolate(frame, [0, 24, 84, 112], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const sceneStart = 105;
  const sceneDuration = Math.floor((durationInFrames - sceneStart - 36) / scenes.length);

  return (
    <AbsoluteFill style={styles.stage}>
      <MovingBackdrop accent={accent} />
      <AbsoluteFill style={styles.vignette} />

      <Sequence durationInFrames={sceneStart}>
        <AbsoluteFill style={{...styles.center, opacity: introOpacity}}>
          <div style={styles.kicker}>programmatic video pipeline</div>
          <h1 style={styles.title}>{title}</h1>
          <p style={styles.subtitle}>{subtitle}</p>
        </AbsoluteFill>
      </Sequence>

      {scenes.map((scene, index) => (
        <Sequence
          key={scene.label}
          from={sceneStart + index * sceneDuration}
          durationInFrames={sceneDuration + 18}
        >
          <SceneCard
            accent={accent}
            index={index}
            scene={scene}
            sceneDuration={sceneDuration}
          />
        </Sequence>
      ))}

      <AbsoluteFill style={styles.footerBar}>
        <span>{footer}</span>
        <span>{Math.round(durationInFrames / fps)}s / {fps}fps</span>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const MovingBackdrop = ({accent}: {accent: string}) => {
  const frame = useCurrentFrame();
  const shift = interpolate(frame, [0, 540], [0, 1], {
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        background:
          `linear-gradient(135deg, #071018 0%, #10262f ${38 + shift * 10}%, #1d1f2a 100%)`,
      }}
    >
      <div
        style={{
          ...styles.mesh,
          background:
            `radial-gradient(circle at ${24 + shift * 30}% 24%, ${accent}55, transparent 28%), ` +
            'radial-gradient(circle at 82% 68%, #f5c84b44, transparent 24%), ' +
            'linear-gradient(120deg, rgba(255,255,255,0.08), transparent 55%)',
        }}
      />
    </AbsoluteFill>
  );
};

const SceneCard = ({
  scene,
  index,
  accent,
  sceneDuration,
}: {
  scene: StoryScene;
  index: number;
  accent: string;
  sceneDuration: number;
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 18, stiffness: 110}});
  const imageScale = interpolate(frame, [0, sceneDuration], [1.08, 1.0], {
    extrapolateRight: 'clamp',
  });
  const exit = interpolate(frame, [sceneDuration - 24, sceneDuration], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        ...styles.sceneShell,
        opacity: exit,
        transform: `translateY(${(1 - enter) * 44}px)`,
      }}
    >
      <div style={styles.visualPanel}>
        {scene.image ? (
          <Img
            src={staticFile(scene.image)}
            style={{
              ...styles.sceneImage,
              transform: `scale(${imageScale})`,
            }}
          />
        ) : (
          <div style={styles.placeholder}>
            <div style={{...styles.placeholderGrid, borderColor: `${accent}66`}} />
            <div style={{...styles.placeholderIndex, color: accent}}>
              {String(index + 1).padStart(2, '0')}
            </div>
          </div>
        )}
      </div>

      <div style={styles.copyPanel}>
        <div style={{...styles.sceneLabel, color: accent}}>{scene.label}</div>
        <h2 style={styles.sceneHeadline}>{scene.headline}</h2>
        <p style={styles.sceneDetail}>{scene.detail}</p>
      </div>
    </AbsoluteFill>
  );
};

const styles: Record<string, React.CSSProperties> = {
  stage: {
    fontFamily: fontStack,
    color: '#f7fbff',
    overflow: 'hidden',
  },
  mesh: {
    position: 'absolute',
    inset: 0,
    filter: 'saturate(1.15)',
  },
  vignette: {
    boxShadow: 'inset 0 0 220px rgba(0,0,0,0.55)',
  },
  center: {
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    padding: 120,
  },
  kicker: {
    color: '#9bb3c7',
    fontSize: 30,
    fontWeight: 700,
    letterSpacing: 0,
    textTransform: 'uppercase',
    marginBottom: 28,
  },
  title: {
    fontSize: 132,
    lineHeight: 0.98,
    margin: 0,
    maxWidth: 1320,
    fontWeight: 860,
  },
  subtitle: {
    fontSize: 42,
    lineHeight: 1.25,
    maxWidth: 900,
    marginTop: 34,
    color: '#c6d4df',
  },
  sceneShell: {
    display: 'grid',
    gridTemplateColumns: '1.08fr 0.92fr',
    gap: 56,
    padding: '96px 112px 120px',
    alignItems: 'center',
  },
  visualPanel: {
    height: 720,
    borderRadius: 8,
    overflow: 'hidden',
    background: '#111b24',
    boxShadow: '0 26px 90px rgba(0,0,0,0.36)',
  },
  sceneImage: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  placeholder: {
    position: 'relative',
    width: '100%',
    height: '100%',
    background:
      'linear-gradient(135deg, rgba(255,255,255,0.12), transparent 35%), linear-gradient(160deg, #14313a, #1f2430)',
  },
  placeholderGrid: {
    position: 'absolute',
    inset: 42,
    border: '2px solid',
    borderRadius: 8,
    backgroundImage:
      'linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px)',
    backgroundSize: '64px 64px',
  },
  placeholderIndex: {
    position: 'absolute',
    left: 68,
    bottom: 56,
    fontSize: 156,
    fontWeight: 850,
    lineHeight: 0.9,
  },
  copyPanel: {
    paddingRight: 32,
  },
  sceneLabel: {
    fontSize: 30,
    fontWeight: 800,
    textTransform: 'uppercase',
    marginBottom: 26,
  },
  sceneHeadline: {
    fontSize: 88,
    lineHeight: 0.98,
    margin: 0,
    fontWeight: 860,
  },
  sceneDetail: {
    fontSize: 34,
    lineHeight: 1.34,
    color: '#c4d2df',
    marginTop: 30,
  },
  footerBar: {
    top: 'auto',
    height: 70,
    padding: '0 112px',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: 22,
    color: '#a4b7c6',
    background: 'rgba(2,8,13,0.4)',
  },
};
