"""Edge Robot — 全功能边缘机器人模式.

Extends voice with hardware control (motors, LED).
Deploy on S100P with robot_mode=True.

Usage::

    python -m askme.blueprints.edge_robot
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import ControlModule, LEDModule

from askme.blueprints.voice import voice

edge_robot = (
    voice
    + Runtime.use(ControlModule)
    + Runtime.use(LEDModule)
)

__all__ = ["edge_robot"]

if __name__ == "__main__":
    import asyncio
    import signal

    from askme.config import load_config

    async def main():
        cfg = load_config()
        app = await edge_robot.build(cfg)

        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)

        await app.start()
        print(f"Edge Robot running — {len(app.modules)} modules")
        print(f"Wired: {len(app.wired_ports)} ports")
        await stop.wait()
        await app.stop()

    asyncio.run(main())
