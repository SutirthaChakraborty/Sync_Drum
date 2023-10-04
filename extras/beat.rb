live_loop :foo do
  b = sync "/osc*/live_loop/foo"
  bpm_value = b[0] > 0 ? b[0] : 120
  
  use_bpm bpm_value
  
  sample :drum_heavy_kick
  sleep 1
  sample :drum_snare_hard
  sleep 1
end
