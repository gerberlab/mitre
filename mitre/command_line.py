""" 
Entry point for running MITRE from the command line.

Calls mitre.main.run_from_config_file on its first argument; we set this
up using argparse so that a sensible usage message can be displayed.

"""

import os
import argparse
import mitre.main

def apparent_file(filename):
    """ Check that the named file exists. """
    # Yes, it would be better style to open the file and
    # pass the handle to run_from_config file to avoid the possibility
    # of deletion between the check and the open operation, but this
    # is unlikely to be a major issue in practice in this case.
    if os.path.exists(filename):
        return filename
    else:
        raise argparse.ArgumentTypeError('Configuration file %s does not exist.' % filename)
    

def run():
    parser = argparse.ArgumentParser(
        description='Launch MITRE calculations as specified in configuration file.',
        epilog='See MITRE documentation for details on available operations and '
               'how to specify them in the configuration file format.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('config_file', help='configuration file to read', nargs='?', type=apparent_file)
    group.add_argument('-t', '--test',
                        action='store_const',
                        dest='action',
                        default='run',
                        const='test',
                        help=('run a short test problem instead of'
                              ' a user-specified configuration file')
                        )
    args = parser.parse_args()
    if args.action == 'run':
        mitre.main.run_from_config_file(args.config_file)
    elif args.action == 'test':
        print 'Launching test problem...'
        test_filename = os.path.join(os.path.dirname(__file__),
                                     'quick_test.cfg')
        mitre.main.run_from_config_file(test_filename)
        print 'Test problem completed successfully.'
    else:
        # argparse should enforce this for us.
        raise 
