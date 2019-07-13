from setuptools import setup, find_packages


setup(
    name='SensorDatasetCollection',
    version='0.1-alpha',
    packages=find_packages(),
    install_requires=[
        'requests',
        'tqdm',
        'numpy',
        'pandas'
    ],
    url='https://github.com/KawashimaHirotaka/SensorDatasetCollection',
    license='MIT',
    author='Kawashima Hirotaka',
    author_email='',
    description='Sensor dataset collection for machine learning',
    long_description=open('README.md').read(),
)
