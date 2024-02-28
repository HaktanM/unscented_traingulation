from setuptools import setup
setup(
    name='unscented triangulation',
    # version='1.0',
    # author='Haktan YALÇIN',
    # description='Globally optimal robust triangulation using semidefinite relaxations',
    # url='https://github.com/Linusnie/robust-triangulation-relaxations',
    keywords='triangulation,',
    python_requires='>=3.10',
    packages=['simulator,visualizer,solver'],
    install_requires=[
        'pandas',
        'tqdm',
        'numba',
        'cvxpy',
        'mosek',
        'tyro',
    ],
)