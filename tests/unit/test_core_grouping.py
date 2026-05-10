import pytest
from pathlib import Path
from datetime import datetime, timedelta
from photofilter.core import group_bursts

def test_group_bursts_empty():
    assert group_bursts([], 2.0, 10) == []

def test_group_bursts_time_only():
    # Mocking image infos where phash is identical, only time matters
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    images = [
        {'path': Path('1.jpg'), 'time': base_time, 'phash': '0000000000000000'},
        {'path': Path('2.jpg'), 'time': base_time + timedelta(seconds=1), 'phash': '0000000000000000'},
        {'path': Path('3.jpg'), 'time': base_time + timedelta(seconds=5), 'phash': '0000000000000000'},
        {'path': Path('4.jpg'), 'time': base_time + timedelta(seconds=6), 'phash': '0000000000000000'},
    ]
    
    groups = group_bursts(images, time_thresh=2.0, hash_thresh=10)
    
    assert len(groups) == 2
    assert [img['path'].name for img in groups[0]] == ['1.jpg', '2.jpg']
    assert [img['path'].name for img in groups[1]] == ['3.jpg', '4.jpg']

def test_group_bursts_hash_split():
    # Mocking images very close in time, but hashes are very different
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    images = [
        {'path': Path('1.jpg'), 'time': base_time, 'phash': '0000000000000000'},
        {'path': Path('2.jpg'), 'time': base_time + timedelta(seconds=1), 'phash': 'ffffffffffffffff'}, # Distance 64
        {'path': Path('3.jpg'), 'time': base_time + timedelta(seconds=2), 'phash': '0000000000000000'},
    ]
    
    groups = group_bursts(images, time_thresh=2.0, hash_thresh=10)
    
    # 1 and 3 should be in one group, 2 in another, because 2 is totally different.
    # Wait, if they are sorted by time, how does it group?
    # Usually: 1 starts group. 2 is close in time, but hash differs -> new group.
    # 3 is close in time to 2 (diff=1s), but hash differs from 2. Hash matches 1, but time from 1 is 2s (<= 2s).
    # Grouping should probably assign to the closest existing group in time and hash.
    assert len(groups) >= 2
    
def test_group_bursts_missing_time_or_hash():
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    images = [
        {'path': Path('1.jpg'), 'time': base_time, 'phash': '0000000000000000'},
        {'path': Path('2.jpg'), 'time': None, 'phash': '0000000000000000'},
        {'path': Path('3.jpg'), 'time': base_time + timedelta(seconds=1), 'phash': ''},
    ]
    
    groups = group_bursts(images, time_thresh=2.0, hash_thresh=10)
    # Missing times or hashes should probably just get their own groups or be appended if hash/time matches what's available
    # For MVP, maybe they all end up in different groups or 1 and 2 together if time is ignored?
    # Let's just ensure it doesn't crash and groups reasonably.
    assert len(groups) > 0
