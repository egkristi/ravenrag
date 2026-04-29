from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ravenrag",
    version="0.1.0",
    author="Erling Kristiansen",
    author_email="egkristi@gmail.com",
    description="Lightweight RAG (Retrieval-Augmented Generation) for local documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/egkristi/ravenrag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],
    },
)
