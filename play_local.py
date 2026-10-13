"""Dedicated local viewer, separate from the user's existing play process."""
import sys
import viser
original = viser.ViserServer
class LocalServer(original):
    def __init__(self, *args, **kwargs):
        kwargs.update(host='127.0.0.1', port=8093)
        super().__init__(*args, **kwargs)
viser.ViserServer = LocalServer
import mjlab_microduck.tasks
from mjlab.scripts.play import run_play, PlayConfig
run_play(sys.argv[1], PlayConfig(checkpoint_file=sys.argv[2], num_envs=1, viewer='viser', device='cpu'))