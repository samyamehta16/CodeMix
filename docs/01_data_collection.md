# Data Collection — YouTube Comments (Tamil-English Code-Mixed)

## 1. Objective

Collect a raw corpus of Tamil-English code-mixed comments from a public
YouTube video's comment section, to serve as the base corpus for the
language-annotation and aggression-detection pipeline described in
`02_annotation_methodology.md`.

## 2. Source

- **Platform:** YouTube
- **API used:** YouTube Data API v3 (`commentThreads.list` endpoint)
- **Video(s) sampled:** 1 video (`video_id = GXpkOXszuIg`) — a Tamil-language
  YouTube video whose comment section naturally contains Tamil-English
  code-mixed conversational text (a common pattern in Tamil YouTube/social
  media comments: viewers switch between Tamil script, romanized Tamil, and
  English within the same sentence).
- **Comments collected:** 1000 (raw, before cleaning)

> If more than one video was actually queried, list each `video_id` here
> with a one-line description of the channel/content type — this matters
> for your report because it affects how representative/diverse the sample
> is (single-video sampling risks topic/audience bias).

## 3. Collection method

1. **API key / auth:** A Google Cloud project was created and the
   **YouTube Data API v3** was enabled for it. An API key (not OAuth, since
   only public read access is needed) was generated from
   `Google Cloud Console -> APIs & Services -> Credentials`.
2. **Endpoint:** `GET https://www.googleapis.com/youtube/v3/commentThreads`
   called with parameters:
   - `part=snippet`
   - `videoId=<target video id>`
   - `maxResults=100` (API maximum per page)
   - `textFormat=plainText` (avoids HTML-escaped text like `&amp;`)
   - `order=time` or `relevance` (whichever was used — confirm and note here)
   - `pageToken=<nextPageToken>` for pagination, repeated until either the
     `nextPageToken` was exhausted or ~1000 comments had been collected.
3. **Field extraction:** for each item in the response, the following
   fields were pulled out of
   `item.snippet.topLevelComment.snippet` and flattened into one CSV row:

   | CSV column      | Source field (API response)                                   |
   |-----------------|----------------------------------------------------------------|
   | `video_id`      | request parameter (`videoId`)                                  |
   | `comment_id`    | `item.snippet.topLevelComment.id`                               |
   | `text`          | `snippet.textOriginal` (or `textDisplay`, plain-text form)      |
   | `author`        | `snippet.authorDisplayName`                                     |
   | `like_count`    | `snippet.likeCount`                                              |
   | `published_at`  | `snippet.publishedAt` (ISO-8601 UTC timestamp)                  |

   Only **top-level comments** were collected (replies within threads were
   not flattened in — confirm and note here if replies were also included).
4. **Output:** rows were written to a single CSV file,
   `data/raw/youtube_comments.csv`, with the header
   `video_id,comment_id,text,author,like_count,published_at`.

## 4. Known limitations of the raw data

- **Single-video sample** → limited topic diversity and possibly limited
  speaker diversity (many comments came from the same audience/community).
- **No content filtering at collection time** — the raw file contains
  duplicates, spam/copy-paste comments, emoji-only comments, and
  non-Tamil/non-English comments; these are removed in the preprocessing
  step (see `02_annotation_methodology.md`, Step 1).
- **Public API quota** — the YouTube Data API v3 free quota is 10,000
  units/day; a `commentThreads.list` call costs 1 unit, so collecting
  ~1000 comments (≈10 paginated calls) is cheap, but this caps how much
  more data can be pulled per day if the dataset is later expanded.
- **No demographic/consent metadata** — only public, already-published
  comment text and public display names were collected; no private user
  data was accessed. `author` here is the public YouTube display name,
  not a verified real identity.

## 5. Reproducing / extending the collection

To collect more comments (e.g. from additional videos, for a larger or
more diverse dataset), the same method can be repeated for other
`video_id`s and the resulting CSVs concatenated (keeping the same 6-column
schema) before running `src/preprocess.py`. If you want, this can be
turned into a small reusable `src/fetch_youtube_comments.py` script here
so future collection runs are one command instead of manual API calls —
say the word and this can be added next.
