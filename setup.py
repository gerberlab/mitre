from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import pkg_resources

USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension("efficient_likelihoods",
              ["mitre/efficient_likelihoods" + ext],
              include_dirs=[])
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

def readme():
    with open('README.md') as f:
        return f.read()

####
# Subclass build_ext so that we can avoid trying to
# access numpy.h until numpy has been installed.
# Code from pandas setup.py
class BuildExt(build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')
        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        build_ext.build_extensions(self)

classifiers= [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License (GPL)",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    ]
    
setup(
    name='mitre',
    version='0.9beta2',
    description='Microbiome Interpretable Temporal Rule Engine',
    long_description=readme(),
    url='http://github.com/gerberlab/mitre',
    author='Eli Bogart',
    author_email='ebogart@bwh.harvard.edu',
    license='GPLv3',
    install_requires = [
        'numpy', 
        'scipy>=0.17.1',
        'pandas>0.20',
        'matplotlib',
        'ete3',
        'pypolyagamma',
        'scikit-learn',
        'tqdm'
    ],
    packages=['mitre','mitre.data_processing','mitre.load_data',
              'mitre.trees', 'mitre.comparison_methods'],
    ext_modules = extensions,
    include_package_data=True,
    entry_points = {'console_scripts': ['mitre=mitre.command_line:run']},
    zip_safe=False,
    cmdclass = {'build_ext': BuildExt},
    classifiers = classifiers,
    keywords = 'microbiome time-series bayesian-inference'
) 
