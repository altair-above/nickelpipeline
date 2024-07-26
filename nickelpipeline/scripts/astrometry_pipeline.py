"""
Perform astrometric calibration on reduced images
""" 

from nickelpipeline.scripts import scriptbase
from nickelpipeline.convenience.log import adjust_global_logger
import logging

class AstrometryPipeline(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(description='Performs astrometric calibration on reduced images', width=width)
        parser.add_argument('reddir', type=str,
                            help='Path to directory with reduced files to astrometrically calibrate.')
        parser.add_argument('-out', '--output_dir', default=None, type=str,
                            help='Path to directory to save calibrated images. Defaults to /astrometric/ in same directory as reddir')
        parser.add_argument('-t', '--output_type', default='image', type=str,
                            help='Whether to return paths to calibrated image or source table.',
                            choices=['image', 'corr'])
        parser.add_argument('-r', '--resolve', action='store_true', 
                            help="re-solves images with previously generated local solves")
        parser.add_argument('-vv', '--very_verbose', action='store_true', 
                            help="Display most detailed logs (use --verbosity for finer control)")
        parser.add_argument('--verbosity', default=4, type=str,
                            help='Level of verbosity to display (5=highest); overrides --verbose', 
                            choices=[1,2,3,4,5])
        return parser

    @staticmethod
    def main(args):
        
        from pathlib import Path
        from nickelpipeline.astrometry.astrometry_api import run_astrometry
        from nickelpipeline.convenience.dir_nav import unzip_directories
        
        if args.very_verbose:
            args.verbosity = 5
        log_levels = {1:'CRITICAL', 2:'ERROR', 3:'WARNING', 4:'INFO', 5:'DEBUG'}
        adjust_global_logger(log_levels[args.verbosity], __name__)
        logger = logging.getLogger(__name__)
              
        logger.debug(f"Extracting images from {args.reddir}")
        dirs = [dir for dir in Path(args.reddir).iterdir()]
        red_files = unzip_directories(dirs, output_format='Path')
        if args.output_dir is None:
            output_dir = str(red_files[0].parent.parent.parent / 'astrometric')
        
        logger.debug(f"Calling run_astrometry()")
        calib_files = run_astrometry(red_files, output_dir, 
                                     mode=args.output_type, )
        
        return calib_files
        
        