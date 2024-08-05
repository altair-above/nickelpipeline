"""
Perform photometric calibration on reduced images
"""

from nickelpipeline.scripts import scriptbase
from nickelpipeline.convenience.log import adjust_global_logger
import logging

class PhotometryPipeline(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(description='Extracts sources from images & stores data in a table', width=width)
        parser.add_argument('reddir', type=str,
                            help='Path to directory with reduced files to photometrically analyze.')
        parser.add_argument('-out', '--output_dir', default=None, type=str,
                            help='Path to directory to save results. Defaults to /photometric/ in same directory as reddir')
        parser.add_argument('-t', '--thresh', default=8.0, type=float,
                            help='Threshold for source detection = background std * thresh.')
        parser.add_argument('-g', '--group', action='store_true', 
                            help='Consolidates groups of sources detected together into one source')
        parser.add_argument('-m', '--mode', default='all', type=str,
                            help='Mode to run photutils PSFPhotometry. `all` recommended.',
                            choices=['all', 'new'])
        parser.add_argument('-f', '--fittype', default='circ', type=str,
                            help='Which type of Moffat fit to use for PSF photometry',
                            choices=['circ', 'ellip'])
        parser.add_argument('-pf', '--plot_final', action='store_true', 
                            help="Displays images with sources & flux labelled")
        parser.add_argument('-pi', '--plot_inters', action='store_true', 
                            help="Displays images with initial sources & source groups for inspection")
        parser.add_argument('-vv', '--very_verbose', action='store_true', 
                            help="Display most detailed logs (use --verbosity for finer control)")
        parser.add_argument('--verbosity', default=4, type=str,
                            help='Level of verbosity to display (5=highest); overrides --verbose', 
                            choices=[1,2,3,4,5])
        return parser

    @staticmethod
    def main(args):
        
        from pathlib import Path

        from nickelpipeline.photometry.psf_photometry import psf_analysis, consolidate_groups
        from nickelpipeline.photometry.aperture_photometry import aperture_analysis
        from nickelpipeline.convenience.dir_nav import unzip_directories

        if args.very_verbose:
            args.verbosity = 5
        log_levels = {1:'CRITICAL', 2:'ERROR', 3:'WARNING', 4:'INFO', 5:'DEBUG'}
        adjust_global_logger(log_levels[args.verbosity], __name__)
        logger = logging.getLogger(__name__)
              
        logger.debug(f"Extracting images from {args.reddir}")
        dirs = list(Path(args.reddir).iterdir())
        # red_files = unzip_directories(dirs, output_format='Path')
        if args.output_dir is None:
            output_dir = dirs[0].parent.parent / 'photometric'
            Path.mkdir(output_dir, exist_ok=True)
        unconsol_dir = output_dir / 'unconsolidated'
        consol_dir = output_dir / 'consolidated'
        Path.mkdir(unconsol_dir, exist_ok=True)
        Path.mkdir(consol_dir, exist_ok=True)
        
        source_catalogs = []
        for obj_dir in dirs:
            if args.group:
                output_dir = consol_dir / obj_dir.name
                Path.mkdir(consol_dir / obj_dir.name, exist_ok=True)
            else:
                output_dir = unconsol_dir / obj_dir.name
                Path.mkdir(unconsol_dir / obj_dir.name, exist_ok=True)
                
            for file in obj_dir.iterdir():
                psf_data = psf_analysis(file, thresh=args.thresh, 
                                        mode=args.mode, fittype=args.fittype, 
                                        plot_final=args.plot_final, 
                                        plot_inters=args.plot_inters,)
                
                filestem = file.stem.split('_')[0]
                if args.group:
                    psf_data = consolidate_groups(psf_data)
                    output_file = output_dir / f'{filestem}_photsrcs_consol.csv'
                else:
                    output_file = output_dir / f'{filestem}_photsrcs.csv'
                
                all_data = aperture_analysis(psf_data, file)
                source_catalogs.append(all_data)
                all_data.write(output_file, format='csv', overwrite=True)  
                
        # for file in red_files:
        #     psf_data = psf_analysis(file, thresh=args.thresh, 
        #                             mode=args.mode, fittype=args.fittype, 
        #                             plot_final=args.plot_final, 
        #                             plot_inters=args.plot_inters,)
        #     if args.group:
        #         psf_data = consolidate_groups(psf_data)
        #     all_data = aperture_analysis(psf_data, file)
            
        #     source_catalogs.append(all_data)
        #     filestem = file.stem.split('_')[0]
        #     if args.group:
        #         output_file = consol_dir / f'{filestem}_photsrcs_consolidated.csv'
        #     else:
        #         output_file = unconsol_dir / f'{filestem}_photsrcs.csv'
        #     all_data.write(output_file, format='csv', overwrite=True)  
        
        return source_catalogs
        
        
        