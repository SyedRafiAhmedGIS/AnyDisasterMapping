UrbanSARFloods test split lists
===============================

- test_weihui.txt (2333), test_nova.txt (2765), test_jubba_1.txt (1658) and
  test_jubba_2.txt (1919) are the exact lists evaluated in this artifact:
  upstream order, filtered to the tiles present on local disk after the
  test-only download (8675 tiles total); jubba split by its Jubba_1 / Jubba_2
  tile-ID token. Counts are attested by test_inference.log
  ("Direct eval: samples=...") and match metrics/*.json exactly.
- upstream_test_*.txt are the original benchmark lists, verbatim
  (2475 / 2893 / 3808 IDs); they exceed the evaluated counts because some
  upstream tile IDs had no corresponding tile file locally.
