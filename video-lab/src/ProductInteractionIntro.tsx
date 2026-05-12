import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const font =
  'Inter, "Microsoft YaHei", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

export const ProductInteractionIntro = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cameraX = interpolate(frame, [0, 220, 460, 720], [0, -120, -260, -360], clamp);
  const titleIn = spring({frame: frame - 18, fps, config: {damping: 18, stiffness: 90}});
  const personX = interpolate(frame, [0, 120, 360, 720], [-330, 0, 28, 72], clamp);
  const robotX = interpolate(frame, [30, 170, 380, 720], [340, 36, -22, -68], clamp);
  const wave = Math.sin(frame / 8);
  const scan = interpolate(frame, [210, 280, 360], [0, 1, 0], clamp);
  const resultIn = spring({frame: frame - 335, fps, config: {damping: 20, stiffness: 120}});
  const finalIn = spring({frame: frame - 560, fps, config: {damping: 18, stiffness: 100}});

  return (
    <AbsoluteFill style={styles.stage}>
      <AbsoluteFill style={styles.backdrop}>
        <div style={styles.grid} />
        <div style={{...styles.lightSweep, transform: `translateX(${cameraX * -0.28}px)`}} />
      </AbsoluteFill>

      <div style={{...styles.world, transform: `translateX(${cameraX}px)`}}>
        <LabSet />
        <div style={{...styles.personWrap, transform: `translateX(${personX}px)`}}>
          <Person wave={wave} active={frame > 145 && frame < 300} />
        </div>
        <div style={{...styles.robotWrap, transform: `translateX(${robotX}px)`}}>
          <RobotDog frame={frame} alert={frame > 235 && frame < 390} />
        </div>
        <SignalArc progress={scan} />
      </div>

      <div style={{...styles.heroCopy, opacity: interpolate(frame, [0, 20, 98, 132], [0, 1, 1, 0], clamp), transform: `translateY(${(1 - titleIn) * 22}px)`}}>
        <div style={styles.eyebrow}>NOVA Dog AskMe</div>
        <h1 style={styles.heroTitle}>会听、会看、会反馈的机器狗上层智能体</h1>
      </div>

      <Dialogue
        from={120}
        to={230}
        side="left"
        title="操作员"
        text="NOVA，检查前方设备状态。"
      />
      <Dialogue
        from={245}
        to={355}
        side="right"
        title="机器狗"
        text="已识别仪表与通道，开始巡检记录。"
      />

      <div
        style={{
          ...styles.resultPanel,
          opacity: interpolate(frame, [330, 370, 540, 610], [0, 1, 1, 0], clamp),
          transform: `translateY(${(1 - resultIn) * 36}px)`,
        }}
      >
        <div style={styles.panelHeader}>实时任务面板</div>
        <Metric label="视觉感知" value="设备状态正常" fill={0.86} delay={350} />
        <Metric label="语音交互" value="指令已确认" fill={0.78} delay={370} />
        <Metric label="事件记录" value="生成巡检摘要" fill={0.92} delay={390} />
      </div>

      <div
        style={{
          ...styles.finalCard,
          opacity: interpolate(frame, [548, 590, 720], [0, 1, 1], clamp),
          transform: `translateY(${(1 - finalIn) * 38}px) scale(${0.96 + finalIn * 0.04})`,
        }}
      >
        <div style={styles.finalKicker}>产品介绍样片</div>
        <div style={styles.finalTitle}>人机自然交互 + 机器人现场执行</div>
        <div style={styles.finalBody}>后续可接入 image2 角色图、真实机器狗素材、配音和字幕，升级为更像广告片的连续视频。</div>
      </div>

      <div style={styles.timeline}>
        <div style={{...styles.timelineFill, width: `${(frame / 720) * 100}%`}} />
      </div>
    </AbsoluteFill>
  );
};

const LabSet = () => {
  return (
    <div style={styles.labSet}>
      <div style={styles.floor} />
      <div style={{...styles.wallPanel, left: 210, top: 226, width: 300, height: 210}} />
      <div style={{...styles.wallPanel, left: 1120, top: 190, width: 360, height: 250}} />
      <div style={styles.deviceTower}>
        <div style={styles.towerLight} />
        <div style={{...styles.towerLight, background: '#31d0aa', top: 78}} />
      </div>
    </div>
  );
};

const Person = ({wave, active}: {wave: number; active: boolean}) => {
  const armRotate = active ? -34 + wave * 8 : -8 + wave * 2;

  return (
    <div style={styles.person}>
      <div style={styles.head} />
      <div style={styles.neck} />
      <div style={styles.body} />
      <div style={{...styles.arm, left: 82, transform: `rotate(${armRotate}deg)`, transformOrigin: '18px 18px'}} />
      <div style={{...styles.arm, right: 82, transform: 'rotate(18deg)', transformOrigin: '18px 18px'}} />
      <div style={{...styles.leg, left: 100, transform: 'rotate(4deg)'}} />
      <div style={{...styles.leg, right: 100, transform: 'rotate(-7deg)'}} />
      <div style={styles.tablet}>
        <div style={styles.tabletGlow} />
      </div>
    </div>
  );
};

const RobotDog = ({frame, alert}: {frame: number; alert: boolean}) => {
  const bob = Math.sin(frame / 6) * 5;
  const legA = Math.sin(frame / 7) * 10;
  const legB = Math.sin(frame / 7 + Math.PI) * 10;

  return (
    <div style={{...styles.robot, transform: `translateY(${bob}px)`}}>
      <div style={styles.robotBody}>
        <div style={{...styles.robotEye, opacity: alert ? 1 : 0.72}} />
      </div>
      <div style={styles.robotHead}>
        <div style={styles.robotLens} />
      </div>
      {[0, 1, 2, 3].map((leg) => (
        <div
          key={leg}
          style={{
            ...styles.robotLeg,
            left: leg < 2 ? 62 + leg * 86 : 250 + (leg - 2) * 82,
            transform: `rotate(${leg % 2 === 0 ? legA : legB}deg)`,
          }}
        >
          <div style={styles.robotFoot} />
        </div>
      ))}
    </div>
  );
};

const SignalArc = ({progress}: {progress: number}) => {
  return (
    <div style={{...styles.signalArc, opacity: progress}}>
      <div style={{...styles.arcRing, transform: `scale(${0.72 + progress * 0.46})`, opacity: 1 - progress * 0.4}} />
      <div style={{...styles.arcRing, transform: `scale(${0.48 + progress * 0.32})`, opacity: 0.9 - progress * 0.3}} />
    </div>
  );
};

const Dialogue = ({
  from,
  to,
  side,
  title,
  text,
}: {
  from: number;
  to: number;
  side: 'left' | 'right';
  title: string;
  text: string;
}) => {
  const frame = useCurrentFrame();
  const shown = interpolate(frame, [from, from + 16, to - 18, to], [0, 1, 1, 0], clamp);

  return (
    <div
      style={{
        ...styles.dialogue,
        left: side === 'left' ? 150 : 'auto',
        right: side === 'right' ? 150 : 'auto',
        opacity: shown,
        transform: `translateY(${(1 - shown) * 24}px)`,
      }}
    >
      <div style={styles.dialogueTitle}>{title}</div>
      <div style={styles.dialogueText}>{text}</div>
    </div>
  );
};

const Metric = ({
  label,
  value,
  fill,
  delay,
}: {
  label: string;
  value: string;
  fill: number;
  delay: number;
}) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [delay, delay + 36], [0, fill], clamp);

  return (
    <div style={styles.metric}>
      <div style={styles.metricTop}>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <div style={styles.track}>
        <div style={{...styles.trackFill, width: `${progress * 100}%`}} />
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  stage: {
    background: '#071018',
    color: '#f5fbff',
    fontFamily: font,
    overflow: 'hidden',
  },
  backdrop: {
    background: 'linear-gradient(135deg, #071018 0%, #0e2230 44%, #181b28 100%)',
  },
  grid: {
    position: 'absolute',
    inset: 0,
    backgroundImage:
      'linear-gradient(rgba(255,255,255,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px)',
    backgroundSize: '72px 72px',
    maskImage: 'linear-gradient(to bottom, transparent, black 16%, black 86%, transparent)',
  },
  lightSweep: {
    position: 'absolute',
    left: 120,
    top: 0,
    width: 980,
    height: 1080,
    background: 'linear-gradient(90deg, transparent, rgba(49,208,170,0.18), transparent)',
    transform: 'skewX(-14deg)',
  },
  world: {
    position: 'absolute',
    left: 160,
    top: 0,
    width: 2300,
    height: 1080,
  },
  labSet: {
    position: 'absolute',
    inset: 0,
  },
  floor: {
    position: 'absolute',
    left: -260,
    right: -260,
    bottom: 0,
    height: 265,
    background: 'linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.36))',
    borderTop: '2px solid rgba(255,255,255,0.12)',
  },
  wallPanel: {
    position: 'absolute',
    border: '1px solid rgba(174,212,235,0.16)',
    borderRadius: 8,
    background: 'rgba(255,255,255,0.035)',
  },
  deviceTower: {
    position: 'absolute',
    left: 1540,
    top: 360,
    width: 150,
    height: 330,
    borderRadius: 8,
    background: 'linear-gradient(180deg, #273544, #111820)',
    boxShadow: '0 24px 70px rgba(0,0,0,0.35)',
  },
  towerLight: {
    position: 'absolute',
    top: 38,
    left: 42,
    width: 66,
    height: 18,
    borderRadius: 8,
    background: '#f5c84b',
    boxShadow: '0 0 24px currentColor',
  },
  personWrap: {
    position: 'absolute',
    left: 560,
    top: 396,
    width: 320,
    height: 450,
  },
  person: {
    position: 'relative',
    width: 320,
    height: 450,
  },
  head: {
    position: 'absolute',
    left: 126,
    top: 0,
    width: 76,
    height: 86,
    borderRadius: '45% 45% 42% 42%',
    background: '#f1c7a8',
  },
  neck: {
    position: 'absolute',
    left: 148,
    top: 76,
    width: 34,
    height: 34,
    background: '#e6b794',
  },
  body: {
    position: 'absolute',
    left: 100,
    top: 102,
    width: 130,
    height: 176,
    borderRadius: 8,
    background: 'linear-gradient(180deg, #2e7fe6, #17436f)',
  },
  arm: {
    position: 'absolute',
    top: 128,
    width: 38,
    height: 162,
    borderRadius: 20,
    background: '#2b6dbb',
  },
  leg: {
    position: 'absolute',
    top: 266,
    width: 44,
    height: 168,
    borderRadius: 18,
    background: '#162235',
    transformOrigin: '22px 20px',
  },
  tablet: {
    position: 'absolute',
    left: 26,
    top: 202,
    width: 96,
    height: 66,
    borderRadius: 8,
    background: '#08131d',
    border: '2px solid rgba(178,223,255,0.42)',
  },
  tabletGlow: {
    position: 'absolute',
    inset: 10,
    borderRadius: 6,
    background: 'linear-gradient(90deg, #31d0aa, #7ac7ff)',
    opacity: 0.65,
  },
  robotWrap: {
    position: 'absolute',
    left: 975,
    top: 558,
    width: 470,
    height: 260,
  },
  robot: {
    position: 'relative',
    width: 470,
    height: 260,
  },
  robotBody: {
    position: 'absolute',
    left: 56,
    top: 48,
    width: 286,
    height: 118,
    borderRadius: 18,
    background: 'linear-gradient(180deg, #dfe8ee, #8394a3)',
    boxShadow: '0 24px 48px rgba(0,0,0,0.25)',
  },
  robotEye: {
    position: 'absolute',
    right: 26,
    top: 38,
    width: 54,
    height: 22,
    borderRadius: 12,
    background: '#31d0aa',
    boxShadow: '0 0 28px #31d0aa',
  },
  robotHead: {
    position: 'absolute',
    left: 318,
    top: 66,
    width: 98,
    height: 78,
    borderRadius: 16,
    background: 'linear-gradient(180deg, #f8fbff, #9daebb)',
  },
  robotLens: {
    position: 'absolute',
    right: 18,
    top: 22,
    width: 28,
    height: 28,
    borderRadius: 20,
    background: '#0a1720',
    border: '5px solid #31d0aa',
  },
  robotLeg: {
    position: 'absolute',
    top: 146,
    width: 24,
    height: 92,
    borderRadius: 12,
    background: '#aebbc5',
    transformOrigin: '12px 8px',
  },
  robotFoot: {
    position: 'absolute',
    left: -20,
    bottom: -10,
    width: 70,
    height: 22,
    borderRadius: 14,
    background: '#d8e0e6',
  },
  signalArc: {
    position: 'absolute',
    left: 856,
    top: 404,
    width: 430,
    height: 300,
    pointerEvents: 'none',
  },
  arcRing: {
    position: 'absolute',
    inset: 0,
    border: '4px solid rgba(49,208,170,0.58)',
    borderRadius: '50%',
  },
  heroCopy: {
    position: 'absolute',
    left: 110,
    top: 88,
    width: 820,
  },
  eyebrow: {
    color: '#31d0aa',
    fontSize: 34,
    fontWeight: 850,
    marginBottom: 18,
  },
  heroTitle: {
    margin: 0,
    fontSize: 68,
    lineHeight: 1.04,
    maxWidth: 820,
    fontWeight: 860,
  },
  dialogue: {
    position: 'absolute',
    top: 160,
    width: 560,
    borderRadius: 8,
    padding: '26px 30px',
    background: 'rgba(8,19,29,0.82)',
    border: '1px solid rgba(178,223,255,0.25)',
    boxShadow: '0 22px 70px rgba(0,0,0,0.28)',
  },
  dialogueTitle: {
    fontSize: 24,
    color: '#9ecfff',
    fontWeight: 800,
    marginBottom: 12,
  },
  dialogueText: {
    fontSize: 38,
    lineHeight: 1.26,
    fontWeight: 720,
  },
  resultPanel: {
    position: 'absolute',
    right: 116,
    top: 104,
    width: 620,
    borderRadius: 8,
    padding: 34,
    background: 'rgba(8,19,29,0.86)',
    border: '1px solid rgba(49,208,170,0.32)',
    boxShadow: '0 28px 100px rgba(0,0,0,0.35)',
  },
  panelHeader: {
    fontSize: 34,
    fontWeight: 840,
    marginBottom: 26,
  },
  metric: {
    marginBottom: 26,
  },
  metricTop: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: 20,
    fontSize: 25,
    color: '#c8d8e5',
    marginBottom: 10,
  },
  track: {
    height: 14,
    borderRadius: 8,
    overflow: 'hidden',
    background: 'rgba(255,255,255,0.12)',
  },
  trackFill: {
    height: '100%',
    borderRadius: 8,
    background: 'linear-gradient(90deg, #31d0aa, #7ac7ff)',
  },
  finalCard: {
    position: 'absolute',
    left: 210,
    right: 210,
    top: 220,
    minHeight: 420,
    borderRadius: 8,
    padding: '58px 72px',
    background: 'rgba(8,19,29,0.88)',
    border: '1px solid rgba(178,223,255,0.22)',
    boxShadow: '0 35px 120px rgba(0,0,0,0.42)',
  },
  finalKicker: {
    color: '#31d0aa',
    fontSize: 30,
    fontWeight: 820,
    marginBottom: 26,
  },
  finalTitle: {
    fontSize: 76,
    lineHeight: 1.06,
    fontWeight: 870,
    maxWidth: 1120,
  },
  finalBody: {
    color: '#c8d8e5',
    fontSize: 34,
    lineHeight: 1.35,
    maxWidth: 1120,
    marginTop: 32,
  },
  timeline: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: 8,
    background: 'rgba(255,255,255,0.12)',
  },
  timelineFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #31d0aa, #7ac7ff)',
  },
};
