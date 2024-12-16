"""
neuralplexer_benchmarks
"""

from setuptools import find_packages, setup

import versioneer

tests_require = ["pytest", "pytest-asyncio", "pytest-cov"]

setup(
    # Self-descriptive entries which should always be present
    name="neuralplexer_benchmarks",
    author="Iambic Therapeutics",
    author_email="help@iambic.ai",
    url="www.iambic.ai",
    description="For benchmarking NeuralPLexer and related Biomolecular Structure Prediction models.",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD 3-Clause",
    packages=[package for package in find_packages() if package.startswith("neuralplexer_benchmarks")],
    include_package_data=True,
    install_requires=[],  # Not used
    extras_require={
        "tests": tests_require,
    },
    platforms=['Linux',
               'Mac OS-X',
    #            'Unix',
               'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.12",  # Python version restrictions
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "np-bench = neuralplexer_benchmarks.cli:app",
        ],
    },
)
