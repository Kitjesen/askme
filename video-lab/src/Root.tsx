import {Composition} from 'remotion';
import {Image2Storyboard} from './Image2Storyboard';
import {NOVAProductPromo} from './NOVAProductPromo';
import {NOVARealImagePromo} from './NOVARealImagePromo';
import {ProductInteractionIntro} from './ProductInteractionIntro';
import {videoConfig} from './videoConfig';

export const Root = () => {
  return (
    <>
      <Composition
        id="Image2Storyboard"
        component={Image2Storyboard}
        durationInFrames={videoConfig.durationInFrames}
        fps={videoConfig.fps}
        width={videoConfig.width}
        height={videoConfig.height}
        defaultProps={videoConfig.props}
      />
      <Composition
        id="ProductInteractionIntro"
        component={ProductInteractionIntro}
        durationInFrames={720}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="NOVAProductPromo"
        component={NOVAProductPromo}
        durationInFrames={690}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="NOVARealImagePromo"
        component={NOVARealImagePromo}
        durationInFrames={540}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
