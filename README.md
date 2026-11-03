# genai-image-qa

Automated quality checks for AI-generated and retouched product images.

**Problem:** In creative production, generated and retouched images go through
manual review before publishing. Common defects (blur, wrong resolution, dirty
backgrounds) are mechanical to detect but slow to check by eye at scale.

**What this does:** Runs a set of automated quality gates on product images and
flags failures for human review, so reviewers only look at images that need a
human decision.

## Status

🔄 Stage 1: project scaffold + first check (blur detection).

Planned checks: blur · resolution · background cleanliness · (more as needed)
Planned: batch runner · generation/edit API wrapper · FastAPI service.

## Setup

```
pip install -r requirements.txt
```

## Run the tests

```
python -m pytest
```

## Author

Karthick (Karthick006-tech). Built from real creative-production workflow needs.
