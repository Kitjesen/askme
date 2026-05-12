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

const particles = Array.from({length: 42}, (_, index) => ({
  left: (index * 127) % 1900,
  top: 120 + ((index * 83) % 710),
  size: 2 + (index % 5),
  speed: 0.16 + (index % 7) * 0.025,
  opacity: 0.18 + (index % 4) * 0.08,
}));

export const NOVAProductPromo = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cameraPush = interpolate(frame, [0, 180, 430, 690], [0, 1, 1.35, 1.8], clamp);
  const globalFade = interpolate(frame, [0, 28, 660, 690], [0, 1, 1, 0], clamp);
  const sceneShift = interpolate(frame, [0, 210, 420, 690], [0, -130, -260, -420], clamp);
  const robotEnter = spring({frame: frame - 70, fps, config: {damping: 19, stiffness: 92}});
  const operatorEnter = spring({frame: frame - 118, fps, config: {damping: 20, stiffness: 84}});
  const uiIn = spring({frame: frame - 242, fps, config: {damping: 20, stiffness: 110}});
  const finale = spring({frame: frame - 535, fps, config: {damping: 19, stiffness: 96}});

  return (
    <AbsoluteFill style={{...styles.stage, opacity: globalFade}}>
      <CinematicBackground frame={frame} cameraPush={cameraPush} />
      <div style={{...styles.scene, transform: `translateX(${sceneShift}px) scale(${1 + cameraPush * 0.035})`}}>
        <Operator enter={operatorEnter} frame={frame} />
        <RobotHero enter={robotEnter} frame={frame} />
        <InteractionBeam frame={frame} />
      </div>

      <OpeningTitle frame={frame} />
      <VoiceMoment frame={frame} />
      <CapabilityHUD frame={frame} progress={uiIn} />
      <Finale frame={frame} progress={finale} />
      <BottomBrand frame={frame} />
    </AbsoluteFill>
  );
};

const CinematicBackground = ({frame, cameraPush}: {frame: number; cameraPush: number}) => {
  return (
    <AbsoluteFill style={styles.bg}>
      <div style={styles.noise} />
      <div
        style={{
          ...styles.lightCone,
          transform: `translateX(${interpolate(frame, [0, 690], [-90, 120], clamp)}px) skewX(-16deg)`,
        }}
      />
      <div style={{...styles.horizonGlow, transform: `scale(${1 + cameraPush * 0.08})`}} />
      <div style={styles.floorGrid} />
      {particles.map((dot, index) => (
        <div
          key={index}
          style={{
            ...styles.particle,
            left: dot.left,
            top: dot.top + Math.sin(frame * dot.speed + index) * 18,
            width: dot.size,
            height: dot.size,
            opacity: dot.opacity,
          }}
        />
      ))}
      <div style={styles.vignette} />
    </AbsoluteFill>
  );
};

const OpeningTitle = ({frame}: {frame: number}) => {
  const titleIn = spring({frame: frame - 12, fps: 30, config: {damping: 20, stiffness: 92}});
  const opacity = interpolate(frame, [0, 24, 118, 156], [0, 1, 1, 0], clamp);

  return (
    <div
      style={{
        ...styles.opening,
        opacity,
        transform: `translateY(${(1 - titleIn) * 28}px)`,
      }}
    >
      <div style={styles.kicker}>NOVA Dog AskMe</div>
      <div style={styles.openingTitle}>让机器狗拥有现场理解力</div>
      <div style={styles.openingSub}>语音交互、视觉巡检、任务闭环，一条智能体链路完成。</div>
    </div>
  );
};

const Operator = ({enter, frame}: {enter: number; frame: number}) => {
  const breathe = Math.sin(frame / 18) * 5;
  const arm = interpolate(frame, [150, 210, 300, 350], [0, -28, -28, 0], clamp);

  return (
    <div
      style={{
        ...styles.operator,
        transform: `translateX(${-180 + enter * 180}px) translateY(${breathe}px)`,
        opacity: interpolate(enter, [0, 0.6], [0, 1], clamp),
      }}
    >
      <div style={styles.operatorRim} />
      <div style={styles.operatorHead}>
        <div style={styles.operatorVisor} />
      </div>
      <div style={styles.operatorTorso} />
      <div style={{...styles.operatorArm, transform: `rotate(${arm}deg)`}}>
        <div style={styles.wristDevice} />
      </div>
      <div style={{...styles.operatorArmBack, transform: 'rotate(13deg)'}} />
      <div style={styles.operatorLegA} />
      <div style={styles.operatorLegB} />
    </div>
  );
};

const RobotHero = ({enter, frame}: {enter: number; frame: number}) => {
  const bob = Math.sin(frame / 9) * 7;
  const legA = Math.sin(frame / 8) * 13;
  const legB = Math.sin(frame / 8 + Math.PI) * 13;
  const scan = interpolate(frame, [210, 300, 390], [0.3, 1, 0.55], clamp);

  return (
    <div
      style={{
        ...styles.robotHero,
        opacity: interpolate(enter, [0, 0.45], [0, 1], clamp),
        transform: `translateX(${220 - enter * 220}px) translateY(${bob}px)`,
      }}
    >
      <div style={styles.robotShadow} />
      <div style={styles.robotSpine} />
      <div style={styles.robotBodyShell}>
        <div style={styles.robotBodyHighlight} />
        <div style={{...styles.robotStatus, opacity: scan}} />
        <div style={styles.robotLogo}>NOVA</div>
      </div>
      <div style={styles.robotNeck} />
      <div style={styles.robotHeadShell}>
        <div style={{...styles.robotLensBig, boxShadow: `0 0 ${28 + scan * 22}px rgba(57,224,190,0.9)`}} />
        <div style={styles.robotMouthLine} />
      </div>
      {[0, 1, 2, 3].map((leg) => (
        <RobotLeg
          key={leg}
          left={leg < 2 ? 82 + leg * 120 : 340 + (leg - 2) * 110}
          angle={leg % 2 === 0 ? legA : legB}
          rear={leg >= 2}
        />
      ))}
    </div>
  );
};

const RobotLeg = ({left, angle, rear}: {left: number; angle: number; rear: boolean}) => {
  return (
    <div style={{...styles.robotLegUnit, left, opacity: rear ? 0.86 : 1}}>
      <div style={{...styles.upperLeg, transform: `rotate(${angle}deg)`}} />
      <div style={{...styles.lowerLeg, transform: `rotate(${-angle * 0.72}deg)`}}>
        <div style={styles.footPad} />
      </div>
    </div>
  );
};

const InteractionBeam = ({frame}: {frame: number}) => {
  const progress = interpolate(frame, [180, 250, 345, 390], [0, 1, 1, 0], clamp);

  return (
    <div style={{...styles.beamWrap, opacity: progress}}>
      <div style={{...styles.beam, transform: `scaleX(${progress})`}} />
      <div style={{...styles.beamDot, left: `${18 + progress * 64}%`}} />
      <div style={styles.beamLabel}>VOICE COMMAND / VISUAL CONTEXT</div>
    </div>
  );
};

const VoiceMoment = ({frame}: {frame: number}) => {
  const operator = interpolate(frame, [150, 180, 245, 272], [0, 1, 1, 0], clamp);
  const robot = interpolate(frame, [270, 300, 365, 392], [0, 1, 1, 0], clamp);

  return (
    <>
      <SpeechBubble
        opacity={operator}
        x={130}
        y={180}
        title="操作员"
        text="NOVA，巡检这条产线。"
      />
      <SpeechBubble
        opacity={robot}
        x={1160}
        y={186}
        title="AskMe"
        text="收到。已读取现场视觉与任务上下文。"
      />
    </>
  );
};

const SpeechBubble = ({
  opacity,
  x,
  y,
  title,
  text,
}: {
  opacity: number;
  x: number;
  y: number;
  title: string;
  text: string;
}) => {
  return (
    <div
      style={{
        ...styles.speech,
        left: x,
        top: y,
        opacity,
        transform: `translateY(${(1 - opacity) * 22}px)`,
      }}
    >
      <div style={styles.speechTitle}>{title}</div>
      <div style={styles.speechText}>{text}</div>
    </div>
  );
};

const CapabilityHUD = ({frame, progress}: {frame: number; progress: number}) => {
  const opacity = interpolate(frame, [330, 365, 520, 565], [0, 1, 1, 0], clamp);

  return (
    <div
      style={{
        ...styles.hud,
        opacity,
        transform: `translateY(${(1 - progress) * 40}px)`,
      }}
    >
      <div style={styles.hudHeader}>
        <span>FIELD INTELLIGENCE</span>
        <strong>Live</strong>
      </div>
      <HudRow frame={frame} delay={360} label="多模态感知" value="识别设备状态、人员位置、风险区域" />
      <HudRow frame={frame} delay={386} label="自然语言任务" value="把一句话转成巡检路线与动作序列" />
      <HudRow frame={frame} delay={412} label="现场闭环" value="回传证据、异常摘要、可追溯记录" />
    </div>
  );
};

const HudRow = ({
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
  const fill = interpolate(frame, [delay, delay + 38], [0, 1], clamp);

  return (
    <div style={styles.hudRow}>
      <div style={styles.hudRowText}>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <div style={styles.hudTrack}>
        <div style={{...styles.hudFill, width: `${fill * 100}%`}} />
      </div>
    </div>
  );
};

const Finale = ({frame, progress}: {frame: number; progress: number}) => {
  const opacity = interpolate(frame, [535, 575, 690], [0, 1, 1], clamp);

  return (
    <div
      style={{
        ...styles.finale,
        opacity,
        transform: `translateY(${(1 - progress) * 42}px) scale(${0.96 + progress * 0.04})`,
      }}
    >
      <div style={styles.finaleKicker}>NOVA Dog AskMe</div>
      <div style={styles.finaleTitle}>把机器狗从遥控设备，升级为现场智能同事。</div>
      <div style={styles.finaleLine} />
      <div style={styles.finaleBody}>可接入真实机器狗镜头、image2 角色图、TTS 配音与品牌字幕，形成完整产品宣传片流水线。</div>
    </div>
  );
};

const BottomBrand = ({frame}: {frame: number}) => {
  return (
    <div style={styles.bottomBrand}>
      <span>NOVA DOG ASKME</span>
      <div style={styles.bottomTrack}>
        <div style={{...styles.bottomFill, width: `${(frame / 690) * 100}%`}} />
      </div>
      <span>AI ROBOT FIELD AGENT</span>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  stage: {
    fontFamily: font,
    color: '#f5fbff',
    background: '#03070d',
    overflow: 'hidden',
  },
  bg: {
    background:
      'linear-gradient(135deg, #02050a 0%, #07131f 40%, #101521 66%, #06070c 100%)',
  },
  noise: {
    position: 'absolute',
    inset: 0,
    opacity: 0.19,
    backgroundImage:
      'linear-gradient(0deg, rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px)',
    backgroundSize: '3px 3px',
  },
  lightCone: {
    position: 'absolute',
    left: 320,
    top: -90,
    width: 900,
    height: 1160,
    background:
      'linear-gradient(90deg, transparent 0%, rgba(47,211,176,0.22) 43%, rgba(128,190,255,0.15) 57%, transparent 100%)',
    filter: 'blur(12px)',
  },
  horizonGlow: {
    position: 'absolute',
    left: 250,
    right: 250,
    bottom: 162,
    height: 280,
    borderRadius: '50%',
    background:
      'radial-gradient(ellipse at center, rgba(47,211,176,0.28), rgba(60,111,169,0.10) 42%, transparent 70%)',
    filter: 'blur(18px)',
  },
  floorGrid: {
    position: 'absolute',
    left: -160,
    right: -160,
    bottom: 0,
    height: 360,
    backgroundImage:
      'linear-gradient(rgba(116,203,255,0.13) 1px, transparent 1px), linear-gradient(90deg, rgba(116,203,255,0.13) 1px, transparent 1px)',
    backgroundSize: '96px 72px',
    transform: 'perspective(680px) rotateX(62deg)',
    transformOrigin: 'bottom center',
    opacity: 0.55,
  },
  particle: {
    position: 'absolute',
    borderRadius: 20,
    background: '#9fe7ff',
    boxShadow: '0 0 16px rgba(159,231,255,0.85)',
  },
  vignette: {
    position: 'absolute',
    inset: 0,
    boxShadow: 'inset 0 0 240px rgba(0,0,0,0.76)',
  },
  scene: {
    position: 'absolute',
    left: 120,
    top: 0,
    width: 2300,
    height: 1080,
  },
  opening: {
    position: 'absolute',
    left: 112,
    top: 92,
    width: 1000,
  },
  kicker: {
    color: '#39e0be',
    fontSize: 30,
    fontWeight: 860,
    marginBottom: 20,
  },
  openingTitle: {
    fontSize: 88,
    lineHeight: 1.02,
    fontWeight: 900,
    maxWidth: 960,
  },
  openingSub: {
    color: '#b7c8d7',
    fontSize: 32,
    lineHeight: 1.35,
    marginTop: 28,
    maxWidth: 820,
  },
  operator: {
    position: 'absolute',
    left: 250,
    top: 385,
    width: 330,
    height: 520,
  },
  operatorRim: {
    position: 'absolute',
    left: 62,
    top: 18,
    width: 190,
    height: 430,
    borderRadius: 90,
    background: 'linear-gradient(90deg, rgba(57,224,190,0.28), transparent)',
    filter: 'blur(20px)',
  },
  operatorHead: {
    position: 'absolute',
    left: 116,
    top: 0,
    width: 94,
    height: 104,
    borderRadius: '48% 48% 42% 42%',
    background: 'linear-gradient(150deg, #f4c8a5, #9b6755)',
    boxShadow: 'inset -14px -10px 28px rgba(0,0,0,0.24)',
  },
  operatorVisor: {
    position: 'absolute',
    left: 40,
    top: 38,
    width: 52,
    height: 18,
    borderRadius: 10,
    background: '#07131f',
    boxShadow: '0 0 16px rgba(57,224,190,0.75)',
  },
  operatorTorso: {
    position: 'absolute',
    left: 82,
    top: 105,
    width: 168,
    height: 220,
    borderRadius: 18,
    background:
      'linear-gradient(135deg, #24364b 0%, #1d78d3 48%, #0b1e32 100%)',
    boxShadow: 'inset -22px -18px 40px rgba(0,0,0,0.28), 0 30px 70px rgba(0,0,0,0.36)',
  },
  operatorArm: {
    position: 'absolute',
    left: 210,
    top: 142,
    width: 44,
    height: 190,
    borderRadius: 22,
    background: 'linear-gradient(180deg, #2c7bd0, #143a61)',
    transformOrigin: '22px 22px',
  },
  wristDevice: {
    position: 'absolute',
    left: -16,
    bottom: -4,
    width: 76,
    height: 44,
    borderRadius: 8,
    background: '#06111b',
    border: '2px solid rgba(57,224,190,0.58)',
  },
  operatorArmBack: {
    position: 'absolute',
    left: 62,
    top: 152,
    width: 42,
    height: 182,
    borderRadius: 22,
    background: '#19375b',
    transformOrigin: '22px 22px',
  },
  operatorLegA: {
    position: 'absolute',
    left: 104,
    top: 308,
    width: 52,
    height: 196,
    borderRadius: 24,
    background: '#101a2b',
    transform: 'rotate(5deg)',
  },
  operatorLegB: {
    position: 'absolute',
    left: 176,
    top: 306,
    width: 52,
    height: 198,
    borderRadius: 24,
    background: '#0d1625',
    transform: 'rotate(-5deg)',
  },
  robotHero: {
    position: 'absolute',
    left: 740,
    top: 496,
    width: 620,
    height: 360,
  },
  robotShadow: {
    position: 'absolute',
    left: 40,
    right: 20,
    bottom: 20,
    height: 58,
    borderRadius: '50%',
    background: 'rgba(0,0,0,0.44)',
    filter: 'blur(16px)',
  },
  robotSpine: {
    position: 'absolute',
    left: 138,
    top: 58,
    width: 294,
    height: 26,
    borderRadius: 20,
    background: 'linear-gradient(90deg, #728796, #e7f0f6, #7b8f9f)',
  },
  robotBodyShell: {
    position: 'absolute',
    left: 86,
    top: 76,
    width: 360,
    height: 138,
    borderRadius: 28,
    background: 'linear-gradient(145deg, #f6fbff 0%, #a8bac9 42%, #4b6072 100%)',
    boxShadow:
      'inset -28px -28px 56px rgba(32,48,62,0.38), inset 18px 18px 34px rgba(255,255,255,0.62), 0 35px 90px rgba(0,0,0,0.34)',
  },
  robotBodyHighlight: {
    position: 'absolute',
    left: 36,
    top: 22,
    width: 150,
    height: 22,
    borderRadius: 20,
    background: 'rgba(255,255,255,0.55)',
    filter: 'blur(7px)',
  },
  robotStatus: {
    position: 'absolute',
    right: 32,
    top: 48,
    width: 72,
    height: 26,
    borderRadius: 16,
    background: '#39e0be',
    boxShadow: '0 0 28px rgba(57,224,190,0.9)',
  },
  robotLogo: {
    position: 'absolute',
    left: 34,
    bottom: 26,
    color: '#1d3445',
    fontSize: 24,
    fontWeight: 900,
  },
  robotNeck: {
    position: 'absolute',
    left: 430,
    top: 104,
    width: 44,
    height: 46,
    borderRadius: 16,
    background: '#8ca0ad',
  },
  robotHeadShell: {
    position: 'absolute',
    left: 464,
    top: 76,
    width: 142,
    height: 104,
    borderRadius: 24,
    background: 'linear-gradient(145deg, #f9fdff, #91a5b4 70%, #5b6e7e)',
    boxShadow: 'inset 12px 14px 26px rgba(255,255,255,0.52), 0 25px 70px rgba(0,0,0,0.34)',
  },
  robotLensBig: {
    position: 'absolute',
    right: 26,
    top: 26,
    width: 42,
    height: 42,
    borderRadius: 24,
    background: '#06111b',
    border: '8px solid #39e0be',
  },
  robotMouthLine: {
    position: 'absolute',
    left: 24,
    bottom: 24,
    width: 74,
    height: 6,
    borderRadius: 6,
    background: 'rgba(10,24,36,0.5)',
  },
  robotLegUnit: {
    position: 'absolute',
    top: 188,
    width: 58,
    height: 130,
  },
  upperLeg: {
    position: 'absolute',
    left: 16,
    top: 0,
    width: 30,
    height: 86,
    borderRadius: 18,
    background: 'linear-gradient(180deg, #cad7df, #718692)',
    transformOrigin: '15px 8px',
  },
  lowerLeg: {
    position: 'absolute',
    left: 18,
    top: 66,
    width: 28,
    height: 78,
    borderRadius: 16,
    background: 'linear-gradient(180deg, #e7eef3, #8194a1)',
    transformOrigin: '14px 8px',
  },
  footPad: {
    position: 'absolute',
    left: -26,
    bottom: -8,
    width: 86,
    height: 24,
    borderRadius: 16,
    background: 'linear-gradient(180deg, #f2f7fa, #a9bac6)',
  },
  beamWrap: {
    position: 'absolute',
    left: 480,
    top: 360,
    width: 760,
    height: 190,
  },
  beam: {
    position: 'absolute',
    left: 0,
    top: 84,
    width: 760,
    height: 3,
    background:
      'linear-gradient(90deg, rgba(57,224,190,0), rgba(57,224,190,0.95), rgba(116,203,255,0))',
    transformOrigin: 'left center',
    boxShadow: '0 0 28px rgba(57,224,190,0.76)',
  },
  beamDot: {
    position: 'absolute',
    top: 72,
    width: 28,
    height: 28,
    borderRadius: 18,
    background: '#d9fff7',
    boxShadow: '0 0 30px rgba(57,224,190,0.95)',
  },
  beamLabel: {
    position: 'absolute',
    left: 170,
    top: 104,
    color: '#9ed8ff',
    fontSize: 20,
    fontWeight: 820,
  },
  speech: {
    position: 'absolute',
    width: 560,
    borderRadius: 8,
    padding: '24px 28px',
    background: 'rgba(5,13,22,0.78)',
    border: '1px solid rgba(134,210,255,0.26)',
    boxShadow: '0 28px 80px rgba(0,0,0,0.38)',
    backdropFilter: 'blur(12px)',
  },
  speechTitle: {
    color: '#39e0be',
    fontSize: 22,
    fontWeight: 860,
    marginBottom: 10,
  },
  speechText: {
    fontSize: 34,
    lineHeight: 1.27,
    fontWeight: 820,
  },
  hud: {
    position: 'absolute',
    right: 102,
    top: 118,
    width: 690,
    borderRadius: 8,
    padding: '32px 34px 26px',
    background: 'rgba(5,13,22,0.82)',
    border: '1px solid rgba(57,224,190,0.32)',
    boxShadow: '0 34px 110px rgba(0,0,0,0.42)',
    backdropFilter: 'blur(16px)',
  },
  hudHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    color: '#9ed8ff',
    fontSize: 22,
    fontWeight: 850,
    marginBottom: 24,
  },
  hudRow: {
    marginBottom: 24,
  },
  hudRowText: {
    display: 'grid',
    gridTemplateColumns: '180px 1fr',
    gap: 22,
    alignItems: 'baseline',
    fontSize: 23,
    color: '#dbe9f4',
    marginBottom: 10,
  },
  hudTrack: {
    height: 11,
    borderRadius: 8,
    background: 'rgba(255,255,255,0.11)',
    overflow: 'hidden',
  },
  hudFill: {
    height: '100%',
    borderRadius: 8,
    background: 'linear-gradient(90deg, #39e0be, #7cc8ff)',
  },
  finale: {
    position: 'absolute',
    left: 190,
    right: 190,
    top: 205,
    minHeight: 470,
    borderRadius: 8,
    padding: '60px 72px',
    background:
      'linear-gradient(135deg, rgba(5,13,22,0.90), rgba(12,32,46,0.88))',
    border: '1px solid rgba(134,210,255,0.26)',
    boxShadow: '0 42px 140px rgba(0,0,0,0.48)',
    backdropFilter: 'blur(20px)',
  },
  finaleKicker: {
    color: '#39e0be',
    fontSize: 28,
    fontWeight: 880,
    marginBottom: 24,
  },
  finaleTitle: {
    fontSize: 72,
    lineHeight: 1.08,
    fontWeight: 920,
    maxWidth: 1240,
  },
  finaleLine: {
    width: 220,
    height: 4,
    borderRadius: 4,
    background: 'linear-gradient(90deg, #39e0be, #7cc8ff)',
    marginTop: 34,
    marginBottom: 30,
  },
  finaleBody: {
    color: '#c6d8e8',
    fontSize: 31,
    lineHeight: 1.38,
    maxWidth: 1180,
  },
  bottomBrand: {
    position: 'absolute',
    left: 70,
    right: 70,
    bottom: 46,
    display: 'grid',
    gridTemplateColumns: 'auto 1fr auto',
    alignItems: 'center',
    gap: 28,
    color: '#91aabd',
    fontSize: 18,
    fontWeight: 820,
  },
  bottomTrack: {
    height: 3,
    borderRadius: 4,
    background: 'rgba(255,255,255,0.12)',
    overflow: 'hidden',
  },
  bottomFill: {
    height: '100%',
    borderRadius: 4,
    background: 'linear-gradient(90deg, #39e0be, #7cc8ff)',
  },
};
