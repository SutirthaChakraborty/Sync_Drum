live_loop :foo do
  b = sync "/osc*/live_loop/foo"
  sample :drum_heavy_kick
  sleep 1
  sample :drum_snare_hard
  sleep 1
  use_bpm b[0]
end