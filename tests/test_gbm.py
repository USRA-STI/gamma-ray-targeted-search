 
import os
import sys
import runpy
import shutil
import tempfile
import urllib.request

from unittest import  mock, TestCase

test_dir = os.path.dirname(os.path.abspath(__file__))
gts_gbm = runpy.run_path(os.path.join(os.path.dirname(sys.executable), "gts-gbm"))

class GtsGbmTest(TestCase):

    @classmethod
    def setUpClass(cls):
        r""" Function to setup any class globals that you want
        to use across individual member functions """
        cls.orig_dir = os.getcwd()
        cls.tmp_dir = tempfile.mkdtemp()
        os.chdir(cls.tmp_dir)

        cls.gw_fits_file = os.path.join(cls.tmp_dir, 'GW170817.fits.gz')
        url = 'https://dcc.ligo.org/public/0146/G1701985/001/preliminary-LALInference.fits.gz'
        urllib.request.urlretrieve(url, cls.gw_fits_file)

    @classmethod
    def tearDownClass(cls):
        r""" Function to clean up at end of tests """
        # remove the temp directory
        os.chdir(cls.orig_dir)
        shutil.rmtree(cls.tmp_dir)

    def execute_command(self, command, user_input=[]):
        # helper function to setup argv and user input needed to run main
        with mock.patch('sys.argv', command.split()):
            with mock.patch('builtins.input') as mock_input:
                mock_input.side_effect = user_input
                gts_gbm['main']()
                
    def test_data(self):
        self.assertTrue(os.path.exists(self.gw_fits_file))

    def test_grb(self):
        # test the gbm_targeted_search on GRB 170817A
        self.execute_command('gts-gbm --burst-number 170817529 --search-window-width 10.0')

    def test_gw(self):
        # test the gbm_targeted_search on GW170817
        self.execute_command(f'gts-gbm --skymap {self.gw_fits_file} --search-window-width 10.0')
