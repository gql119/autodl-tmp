import pytest

from oa_lgc.episodes import DisjointEpisodeSampler, ImageRecord, preserve_source_id


def _records(count=8):
    records = []
    for index in range(count):
        annotations = [{"cls": 14, "bbox": [0.5, 0.5, 0.4, 0.4]}]
        if index % 2 == 0:
            annotations.append({"cls": 1, "bbox": [0.2, 0.2, 0.1, 0.1]})
        records.append(ImageRecord(f"image_{index}", f"/{index}.jpg", tuple(annotations)))
    return records


def test_support_query_disjoint_ids():
    episode = DisjointEpisodeSampler(_records(), 14, support_size=2, query_size=2).sample()
    assert set(episode.support_ids).isdisjoint(episode.query_ids)


def test_support_query_reproducibility():
    first = DisjointEpisodeSampler(_records(), 14, seed=7).sample(3)
    second = DisjointEpisodeSampler(_records(), 14, seed=7).sample(3)
    changed = DisjointEpisodeSampler(_records(), 14, seed=8).sample(3)
    assert (first.support_ids, first.query_ids) == (second.support_ids, second.query_ids)
    assert (first.support_ids, first.query_ids) != (changed.support_ids, changed.query_ids)


def test_clean_poison_pair_identity():
    episode = DisjointEpisodeSampler(_records(), 14).sample()
    assert [item.source_id for item in episode.support_clean] == [item.source_id for item in episode.support_poison]
    assert [item.source_id for item in episode.query_clean] == [item.source_id for item in episode.query_poison]


def test_insufficient_episode_data_failure():
    sampler = DisjointEpisodeSampler(_records(3), 14, support_size=2, query_size=2)
    with pytest.raises(RuntimeError, match="reuse is forbidden"):
        sampler.sample()


def test_augmentation_preserves_source_id():
    record = _records(1)[0]
    assert preserve_source_id(record, "horizontal_flip") == record.source_id


def test_target_presence_in_support_query():
    episode = DisjointEpisodeSampler(_records(), 14).sample()
    assert all(any(annotation["cls"] == 14 for annotation in record.annotations) for record in episode.support_clean)
    assert all(any(annotation["cls"] == 14 for annotation in record.annotations) for record in episode.query_clean)


def test_class_validity_mask():
    episode = DisjointEpisodeSampler(_records(), 14, support_size=4, query_size=4).sample()
    assert episode.class_validity[14] is False
    assert episode.class_counts[1]["support"] >= 1
    assert episode.class_counts[1]["query"] >= 1
    assert episode.class_validity[1] is True
    assert episode.class_validity[2] is False


def test_multi_worker_episode_internal_disjointness():
    sampler = DisjointEpisodeSampler(_records(20), 14, support_size=3, query_size=3, seed=2)
    for worker_id in range(4):
        episode = sampler.sample(episode_index=1, worker_id=worker_id)
        assert set(episode.support_ids).isdisjoint(episode.query_ids)

