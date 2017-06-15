""" 
Entry point for running MITRE_EVENTS from the command line.

Calls mitre.events.run_from_config_file on its first argument; we set this
up using argparse so that a sensible usage message can be displayed.

"""

import os
import argparse
import mitre.events

def run():
    parser = argparse.ArgumentParser(
        description='Launch MITRE_EVENTS calculations as specified in configuration file.',
        epilog='See MITRE documentation and the draft MITRE_EVENTS writeup for details on available operations and '
               'how to specify them in the configuration file format.')
    parser.add_argument('config_file',type=str,help='configuration file to read')
    args = parser.parse_args()
    mitre.events.run_from_config_file(args.config_file)
