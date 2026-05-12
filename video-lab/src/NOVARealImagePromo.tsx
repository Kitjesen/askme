import React from 'react';
import {
  AbsoluteFill,
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const font =
  'Inter, "Microsoft YaHei", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

export const NOVARealImagePromo = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const intro = spring({frame: frame - 10, fps, config: {damping: 20, stiffness: 90}});
  const hud = spring({frame: frame - 150, fps, config: {damping: 18, stiffness: 105}});
  const finale = spring({frame: frame - 385, fps, config: {damping: 20, stiffness: 95}});
  const globalFade = interpolate(frame, [0, 24, 510, 540], [0, 1, 1, 0], clamp);
  const zoom = interpolate(frame, [0, 190, 370, 540], [1.02, 1.1, 1.17, 1.22], clamp);
  const panX = interpolate(frame, [0, 210, 540], [-38, -116, -182], clamp);
  const panY = interpolate(frame, [0, 260, 540], [0, -22, -38], clamp);

  return (
    <AbsoluteFill style={{...styles.stage, opacity: globalFade}}>
      <AbsoluteFill>
        <Img
          src={staticFile('assets/image2/nova-real-hero.png')}
          style={{
            ...styles.heroImage,
            transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
          }}
        />
        <div style={styles.grade} />
        <div style={styles.vignette} />
      </AbsoluteFill>

      <div
        style={{
          ...styles.titleBlock,
          opacity: interpolate(frame, [0, 28, 116, 150], [0, 1, 1, 0], clamp),
          transform: `translateY(${(1 - intro) * 26}px)`,
        }}
      >
        <div style={styles.kicker}>NOVA Dog AskMe</div>
        <div style={styles.title}>真实现场里的机器狗智能体</div>
        <div style={styles.subtitle}>一句话发起任务，机器狗理解现场并回传证据。</div>
      </div>

      <div
        style={{
          ...styles.voiceCard,
          opacity: interpolate(frame, [115, 142, 235, 270], [0, 1, 1, 0], clamp),
          transform: `translateY(${interpolate(frame, [115, 145], [20, 0], clamp)}px)`,
        }}
      >
        <div style={styles.voiceRole}>操作员</div>
        <div style={styles.voiceText}>NOVA，巡检这条产线。</div>
      </div>

      <div
        style={{
          ...styles.scanLine,
          opacity: interpolate(frame, [150, 190, 315, 350], [0, 1, 1, 0], clamp),
          transform: `translateX(${interpolate(frame, [150, 350], [-120, 320], clamp)}px)`,
        }}
      />

      <div
        style={{
          ...styles.hud,
          opacity: interpolate(frame, [160, 188, 355, 390], [0, 1, 1, 0], clamp),
          transform: `translateY(${(1 - hud) * 34}px)`,
        }}
      >
        <div style={styles.hudHeader}>
          <span>FIELD AGENT ONLINE</span>
          <strong>LIVE</strong>
        </div>
        <HudMetric frame={frame} delay={185} label="视觉理解" value="识别设备、通道、人员" />
        <HudMetric frame={frame} delay={215} label="语音任务" value="自然语言转巡检指令" />
        <HudMetric frame={frame} delay={245} label="闭环反馈" value="证据回传 / 异常摘要" />
      </div>

      <div
        style={{
          ...styles.finalPanel,
          opacity: interpolate(frame, [375, 410, 540], [0, 1, 1], clamp),
          transform: `translateY(${(1 - finale) * 40}px)`,
        }}
      >
        <div style={styles.finalKicker}>NOVA Dog AskMe</div>
        <div style={styles.finalTitle}>让机器狗成为可对话、可观察、可交付的现场智能同事。</div>
        <div style={styles.finalSub}>真实视觉素材 + Remotion 镜头语言 + HUD 产品信息层。</div>
      </div>

      <div style={styles.bottomBar}>
        <span>AI ROBOT FIELD AGENT</span>
        <div style={styles.progressTrack}>
          <div style={{...styles.progressFill, width: `${(frame / 540) * 100}%`}} />
        </div>
        <span>REALISTIC IMAGE2 PROMO</span>
      </div>
    </AbsoluteFill>
  );
};

const HudMetric = ({
  frame,
  delay,
  label,
  value,
}: {
  frame: number;
  delay: number;
  label: string;
  value: string;
}) => {
  const progress = interpolate(frame, [delay, delay + 34], [0, 1], clamp);

  return (
    <div style={styles.metric}>
      <div style={styles.metricTop}>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <div style={styles.metricTrack}>
        <div style={{...styles.metricFill, width: `${progress * 100}%`}} />
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  stage: {
    fontFamily: font,
    color: '#f7fbff',
    background: '#05080d',
    overflow: 'hidden',
  },
  heroImage: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    transformOrigin: '60% 54%',
    filter: 'contrast(1.08) saturate(1.08)',
  },
  grade: {
    position: 'absolute',
    inset: 0,
    background:
      'linear-gradient(90deg, rgba(2,7,12,0.58), rgba(2,7,12,0.12) 42%, rgba(2,7,12,0.46)), linear-gradient(0deg, rgba(2,7,12,0.72), transparent 36%, rgba(2,7,12,0.36))',
  },
  vignette: {
    position: 'absolute',
    inset: 0,
    boxShadow: 'inset 0 0 260px rgba(0,0,0,0.72)',
  },
  titleBlock: {
    position: 'absolute',
    left: 94,
    top: 86,
    width: 860,
  },
  kicker: {
    color: '#39e0be',
    fontSize: 28,
    fontWeight: 880,
    marginBottom: 18,
  },
  title: {
    fontSize: 78,
    lineHeight: 1.04,
    fontWeight: 920,
    maxWidth: 820,
    textShadow: '0 18px 58px rgba(0,0,0,0.56)',
  },
  subtitle: {
    color: '#c4d5e2',
    fontSize: 31,
    lineHeight: 1.36,
    marginTop: 26,
    maxWidth: 720,
  },
  voiceCard: {
    position: 'absolute',
    left: 120,
    bottom: 170,
    width: 620,
    borderRadius: 8,
    padding: '26px 30px',
    background: 'rgba(5,12,20,0.72)',
    border: '1px solid rgba(141,213,255,0.26)',
    boxShadow: '0 28px 90px rgba(0,0,0,0.42)',
    backdropFilter: 'blur(16px)',
  },
  voiceRole: {
    color: '#8bcfff',
    fontSize: 22,
    fontWeight: 880,
    marginBottom: 10,
  },
  voiceText: {
    color: '#f7fbff',
    fontSize: 30,
    lineHeight: 1.25,
    fontWeight: 850,
  },
  scanLine: {
    position: 'absolute',
    left: 760,
    top: 486,
    width: 520,
    height: 3,
    borderRadius: 4,
    background:
      'linear-gradient(90deg, transparent, rgba(57,224,190,0.95), rgba(139,207,255,0.9), transparent)',
    boxShadow: '0 0 34px rgba(57,224,190,0.88)',
  },
  hud: {
    position: 'absolute',
    right: 94,
    top: 110,
    width: 660,
    borderRadius: 8,
    padding: '32px 34px',
    background: 'rgba(5,12,20,0.68)',
    border: '1px solid rgba(57,224,190,0.3)',
    boxShadow: '0 36px 120px rgba(0,0,0,0.46)',
    backdropFilter: 'blur(18px)',
  },
  hudHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    color: '#a7d8ff',
    fontSize: 20,
    fontWeight: 860,
    marginBottom: 26,
  },
  metric: {
    marginBottom: 25,
  },
  metricTop: {
    display: 'grid',
    gridTemplateColumns: '128px 1fr',
    gap: 20,
    alignItems: 'baseline',
    color: '#d7e8f5',
    fontSize: 24,
    marginBottom: 10,
  },
  metricTrack: {
    height: 10,
    borderRadius: 8,
    background: 'rgba(255,255,255,0.13)',
    overflow: 'hidden',
  },
  metricFill: {
    height: '100%',
    borderRadius: 8,
    background: 'linear-gradient(90deg, #39e0be, #8bcfff)',
  },
  finalPanel: {
    position: 'absolute',
    left: 96,
    right: 96,
    bottom: 118,
    borderRadius: 8,
    padding: '42px 52px',
    background:
      'linear-gradient(90deg, rgba(5,12,20,0.84), rgba(5,12,20,0.54))',
    border: '1px solid rgba(141,213,255,0.24)',
    boxShadow: '0 36px 120px rgba(0,0,0,0.44)',
    backdropFilter: 'blur(18px)',
  },
  finalKicker: {
    color: '#39e0be',
    fontSize: 25,
    fontWeight: 880,
    marginBottom: 16,
  },
  finalTitle: {
    fontSize: 58,
    lineHeight: 1.1,
    fontWeight: 920,
    maxWidth: 1350,
  },
  finalSub: {
    color: '#c4d5e2',
    fontSize: 27,
    marginTop: 20,
  },
  bottomBar: {
    position: 'absolute',
    left: 70,
    right: 70,
    bottom: 42,
    display: 'grid',
    gridTemplateColumns: 'auto 1fr auto',
    gap: 26,
    alignItems: 'center',
    color: '#9ab0bf',
    fontSize: 18,
    fontWeight: 840,
  },
  progressTrack: {
    height: 3,
    borderRadius: 4,
    background: 'rgba(255,255,255,0.14)',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #39e0be, #8bcfff)',
  },
};
