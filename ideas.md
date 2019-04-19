# TF World Demo Ideas

## Resume Heat Mapping

Attendee: scans resume in our PDF scanner (or emails) and we manually upload to our server/api
Output: monitor displays heatmap over image of their resume based on the importance of keywords, etc

### Algorithm:

1. **cluster** jobdescriptions into roles (Data Scientist, Web Developer, etc)
2. **distance** from attendee resume to closest category centroid (USE vector)
3. **distance** of each sentence to closest category centroid
4. distance of each word in sentance to category centroid (may be tricky)
5. opencv to display resume image (jpg)
6. opencv to overlay transparent highlighting of high quality words

### Data

- job description (free on [indeed api](https://medium.com/@msalmon00/web-scraping-job-postings-from-indeed-96bd588dcb4b)
- resumes
    - [livecareer.com](https://www.livecareer.com/resume-search/search?jt=data%20science) obfuscated html, scored for quality
    - [jobspider](http://www.jobspider.com/job/resume-search-results.asp/county_San+Diego/miles_10/words_data+scientist/searchtype_1) unlimited HTML `<table>` scraping

### Manceps technologies:

- text extraction from mixed media PDFs and images
- sentence-level modeling of documents (USE)
- document summarization
- document classification
- document visualization (not yet built, but useful for our core product?)
- hybrid ROUGE+USE distance metric


## Dynamic 3D Resume and Job Visualization

[mapper-tda](https://github.com/ksanjeevan/mapper-tda)
[scikit-tda](https://github.com/scikit-tda/kepler-mapper/tree/master/examples)
[word blobs](https://bl.ocks.org/mbostock/1846692)
[companies](http://bl.ocks.org/mbostock/2706022)
[les miserables characters](https://observablehq.com/@d3/force-directed-graph)

- easy to build
- demonstrates Manceps technologies
    - high dimensional clustering
    - high dimensional visualization
    - graph topological data analysis (tda mapping)

## Face Style Transfer

[WarpGAN](https://github.com/seasonSH/WarpGAN)

- caricatures of attendees in front of camera
- morphing of face into other faces

## Find your "dopelfamer" (famous dopelganger)

[Image Similarity Web App](https://github.com/DeepCanopy/image-similarity)

- training data:
    - open source images of famous people faces
- easy to build
- demonstrates Manceps technologies
    - high dimensional clustering
    - high dimensional visualization
    - graph topological data analysis (tda mapping)


## Other [TF Ideas](https://experiments.withgoogle.com/collection/ai)
