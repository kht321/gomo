"""Setup script for Gomoku AI package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A modular Gomoku (Five in a Row) game implementation with AI agents"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            # Only include core dependencies, not optional ones
            lines = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Skip optional dependencies that should be in extras_require
                    if not any(dep in line.lower() for dep in ['transformers', 'torch', 'accelerate', 'pytest', 'black', 'flake8', 'mypy', 'sphinx', 'jupyter', 'matplotlib']):
                        lines.append(line)
            return lines
    return ['openai>=1.0.0']

setup(
    name="gomoku-ai",
    version="2.0.0",
    author="Gomoku AI Team",
    author_email="contact@gomoku-ai.com",
    description="A modular Gomoku (Five in a Row) game implementation with AI agents",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gomoku-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'huggingface': [
            'transformers>=4.20.0',
            'torch>=1.12.0',
            'accelerate>=0.20.0',
        ],
        'dev': [
            'pytest>=6.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'mypy>=1.0',
        ],
        'docs': [
            'sphinx>=5.0',
            'sphinx-rtd-theme>=1.0',
        ],
        'examples': [
            'jupyter>=1.0',
            'matplotlib>=3.5',
        ]
    },
    keywords=[
        'gomoku', 'five-in-a-row', 'board-game', 'ai', 'machine-learning',
        'llm', 'openai', 'huggingface', 'transformers', 'strategy', 'tournament', 'game-ai'
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/gomoku-ai/issues",
        "Source": "https://github.com/your-username/gomoku-ai",
        "Documentation": "https://gomoku-ai.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)