# photo-geolocator-01

make.fun bounty entry: an agent that finds where a photo was taken.

`main.ts` (Node 22, standard library only) fetches the instance's 25 photos
from the trial env, asks a vision model through the gateway
(claude-sonnet-5-5 first, gpt-6.1-sol as fallback) with a prompt that makes it
read the visible clues before guessing, and posts every answer back. A photo
whose model calls all fail gets a fixed prior guess (48.0, 10.0) instead of
being left unanswered.
