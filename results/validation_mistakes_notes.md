# The 13 wrong validation pieces, checked without listening

How each one was checked:
- **Source:** compare the title with the YouTube video the MIDI was transcribed from.
- **Piano check:** GiantMIDI's estimate that the video is solo piano.
- **Render:** look for silence or a broken render.
- **Music:** ask whether the mistake makes musical sense.

None of the renders are broken: silences are a few seconds in each, and the lengths are normal except one.

## Data problems
| Piece | Called | Problem |
|---|---|---|
| Peyer, *Fuga quarti toni* RW 47 (fugue) | waltz, 96% | The MIDI was transcribed from a **trombone quartet**, not piano. The label is right, but the audio is a piano transcription of brass. |
| Méreaux, *Grande valse* Op.46 (waltz) | nocturne, 70% | 26 minutes long, from a "sight read" video. Far longer than a normal waltz, and a sight-reading has unsteady rhythm. |
| Henselt, *Nocturne* Op.27 (nocturne) | waltz, 65% | The video says Op. **32**, so the title or video may be a different piece. Still a nocturne either way. |

**The same problem in training** (found while checking the above):
- 6 more trombone-group fugues (Briegel ×3, Peyer ×2, Martini). That's 6 of 24 training fugues.
- Ropartz, *Fugue in E minor*: the video is "Piece in E flat minor", which looks like a different piece, so probably a **wrong label**.
- Busoni, *Variationen und Fuge*: mostly variations with a fugue at the end. The clips are probably not fugue, the same problem as the excluded "Prelude and Fugue" titles.

## Musically understandable mistakes
- **Nocturne → waltz (5 pieces):** many nocturnes have a rocking left hand in 3s, close to a waltz. This is the model's main confusion.
- **Debussy, *Valse romantique* → nocturne:** a slow, dreamy waltz, close to a nocturne.
- **Bach, *March* BWV Anh.127 → fugue:** written in two independent voices, as a fugue is.
- **Schubert, *Grande marche héroïque* → waltz:** 17 minutes long, with slower middle sections. The three clips may miss the march parts.
- **Simpson, *Fugue* → march**, and **Teilman, *Nocturne* → fugue:** no clear reason.

## Worth listening to (no music knowledge needed)
| File | Listen for |
|---|---|
| `data/audio/midi_render/ad12b5070c32fe0b37a4.flac` (Peyer) | Does it sound like slow, sustained chords, or messy notes, rather than normal piano music? |
| `data/audio/midi_render/8c53a39267dfb9b6453f.flac` (Ropartz) | A fugue starts with one tune alone, then the same tune enters again higher or lower while the first keeps going. Is that what happens, or is it a melody with accompaniment? |
| `data/audio/midi_render/43b7eff7faef5c20a999.flac` (Méreaux) | At 6:30, 13:10 and 19:45, where the clips are taken, can you count "ONE two three"? |
