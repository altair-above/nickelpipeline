"""
Perform reduction of raw astronomical data frames (overscan subtraction,
bias subtraction, flat division, cosmic ray masking)
""" 

from nickelpipeline.scripts import scriptbase


# import pkg_resources
# import json
# import logging.config
# # Load the JSON configuration from a file
# with pkg_resources.resource_stream('nickelpipeline.convenience', 'logging_config.json') as f:
#     config = json.load(f)
#     config['handlers']['file']['filename'] = f"log/{__name__}_script_log.log"
# # Configure logging with the loaded configuration
# logging.config.dictConfig(config)

# import logging
# from nickelpipeline.convenience.log import change_log_file
# logger = logging.getLogger(__name__)
# change_log_file(logger, f"log/{__name__.split('.')[-1]}_script_log.log")


# import logging
# logger = logging.getLogger(__name__)


class ReductionPipeline(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        parser = super().get_parser(description='Reduce images: subtract & trim overscan, subtract bias, divide flat', width=width)
        parser.add_argument('-dir', '--rawdir', default=str(Path().resolve()), type=str,
                            help='Directory with raw files to reduce.')
        parser.add_argument('-fin', '--table_path_in', default='reduction_files_table.yml', type=str,
                            help='Path to input table file with raw FITS file information.')
        parser.add_argument('-fout', '--table_path_out', default=None, type=str,
                            help='Path to output table file for storing the raw FITS file information.')
        parser.add_argument('-s', '--save_inters', default=False, type=bool,
                            help='If True, save intermediate results during processing.')
        parser.add_argument('--excl_files', default=[], type=list,
                            help='List of file stems substrings to exclude (exact match not necessary).')
        parser.add_argument('--excl_obj_strs', default=[], type=list,
                            help='List of object substrings to exclude (exact match not necessary).')
        parser.add_argument('--excl_filts', default=[], type=list,
                            help='List of filter substrings to exclude (exact match not necessary).')
        return parser

    @staticmethod
    def main(args):
        
        from nickelpipeline.reduction.reduction import reduce_all
        from nickelpipeline.convenience.display_fits import display_many_nickel
        
        from nickelpipeline.convenience.log import change_log_file
        import logging
        logger = logging.getLogger(__name__)
        change_log_file(logger, f"log/{__name__.split('.')[-1]}_script_log.log")
        
        
        logger.info("Running reduce_all()")
        redfiles = reduce_all(args.rawdir, args.table_path_in, args.table_path_out,
                              args.save_inters, args.excl_files, args.excl_obj_strs, 
                              args.excl_filts)
        
        display_many_nickel(redfiles)
        
        
        # module_name = __name__.split('.')[-1]
        # logger = default_logger(module_name)
        # logger.info('testing')

# if __name__ == "__main__":
#     ReductionPipeline.entry_point()