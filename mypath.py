
class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            root_dir = '/datasets/UCF101/ufc101'
            output_dir = '/datasets/UCF101/ufc101_preprocessed_128x178'
            # output_dir = '/datasets/UCF101/ufc101_preprocessed_256x256'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
