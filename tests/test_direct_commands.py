"""Tests for direct_commands — predefined robot joint-position poses."""

from __future__ import annotations

import numpy as np

from askme.robot.direct_commands import get_command, list_commands


class TestGetCommand:
    def test_home_returns_array(self):
        cmd = get_command("home")
        assert cmd is not None
        assert cmd.shape == (16,)

    def test_wave_returns_array(self):
        cmd = get_command("wave")
        assert cmd is not None
        assert cmd.shape == (16,)

    def test_grab_returns_array(self):
        cmd = get_command("grab")
        assert cmd is not None
        assert cmd.shape == (16,)

    def test_release_returns_array(self):
        cmd = get_command("release")
        assert cmd is not None

    def test_point_forward_returns_array(self):
        cmd = get_command("point_forward")
        assert cmd is not None

    def test_rest_returns_array(self):
        cmd = get_command("rest")
        assert cmd is not None

    def test_unknown_returns_none(self):
        assert get_command("fly") is None

    def test_empty_string_returns_none(self):
        assert get_command("") is None

    def test_returns_float32(self):
        cmd = get_command("home")
        assert cmd.dtype == np.float32

    def test_returns_copy_not_reference(self):
        """Modifying the returned array should not change the stored command."""
        cmd1 = get_command("home")
        cmd1[0] = 999.0
        cmd2 = get_command("home")
        assert cmd2[0] != 999.0

    def test_grab_has_nonzero_finger_values(self):
        """grab command should close the fingers (fingers 6-9 = 1.0)."""
        cmd = get_command("grab")
        assert cmd[6] > 0.0
        assert cmd[7] > 0.0

    def test_home_fingers_are_zero(self):
        """home pose should have all fingers open (0.0)."""
        cmd = get_command("home")
        np.testing.assert_array_equal(cmd[6:10], 0.0)


class TestListCommands:
    def test_returns_list(self):
        result = list_commands()
        assert isinstance(result, list)

    def test_contains_standard_commands(self):
        cmds = list_commands()
        assert "home" in cmds
        assert "wave" in cmds
        assert "grab" in cmds
        assert "release" in cmds
        assert "rest" in cmds

    def test_is_sorted(self):
        cmds = list_commands()
        assert cmds == sorted(cmds)

    def test_all_commands_retrievable(self):
        for name in list_commands():
            assert get_command(name) is not None
