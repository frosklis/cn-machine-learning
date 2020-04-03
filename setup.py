import glob
import os
import shutil

from setuptools import setup, Command

from cnml._version import __version__

project = "cnml"


class CompleteClean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        shutil.rmtree('./build', ignore_errors=True)
        shutil.rmtree('./dist', ignore_errors=True)
        shutil.rmtree('./' + project + '.egg-info', ignore_errors=True)
        temporal = glob.glob('./' + project + '/*.pyc')
        for t in temporal:
            os.remove(t)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name=project,
    python_requires='>=3',
    version=__version__,
    description="Machine learning algorithm implementations",
    long_description=read('README.rst'),
    license=open('LICENSE'),
    author='Claudio Noguera',
    author_email='claudio.noguera@gmail.com',
    url='https://github.com/frosklis/cn-machine-learning',
    packages=[project, 'tests'],
    install_requires=requirements,
    cmdclass={'clean': CompleteClean},
)
