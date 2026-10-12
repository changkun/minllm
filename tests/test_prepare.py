import gzip

import numpy as np

from minllm import prepare

STORY = [f"On day {i} the fox ran over the hill to the old barn, and it counted the hens that it had seen there before." for i in range(1, 7)]


def record(kind, url, language, text):
    body = text.encode()
    head = f"WARC/1.0\r\nWARC-Type: {kind}\r\nWARC-Target-URI: {url}\r\n" + (f"WARC-Identified-Content-Language: {language}\r\n" if language else "") + f"Content-Length: {len(body)}\r\n\r\n"
    return head.encode() + body + b"\r\n\r\n"


def test_pages_reads_records_by_their_length(tmp_path):
    path = tmp_path / "sample.warc.wet.gz"
    tricky = "A page that quotes a header:\nWARC/1.0\nContent-Length: 3\n\nand goes on."
    with gzip.open(path, "wb") as f:
        f.write(record("warcinfo", "", "", "about this file"))
        f.write(record("conversion", "http://a.example/", "eng,deu", tricky))
        f.write(record("conversion", "http://b.example/", "", "Schöne Grüße."))
    assert list(prepare.pages(path)) == [("http://a.example/", "eng", tricky), ("http://b.example/", "", "Schöne Grüße.")]


def test_prose_keeps_sentences_and_drops_menus():
    page = "Home\nAbout us\nContact\nThe shop opened in 1984 and has sold bread ever since.\nRead more\n“We bake every morning,” she said.\nShare this page on\n© 2024 All rights reserved"
    assert prepare.prose(page) == ["The shop opened in 1984 and has sold bread ever since.", "“We bake every morning,” she said."]


def test_verdict_names_the_first_rule_that_fails():
    assert prepare.verdict(STORY) is None
    assert prepare.verdict(STORY[:2]) == "too short"
    assert prepare.verdict(["Der Fuchs lief über den Hügel zur alten Scheune und zählte dort die Hühner noch einmal."] * 6) == "not English"
    assert prepare.verdict(["这是一个关于狐狸的故事 the fox and the hens 狐狸跑过山丘去旧谷仓数母鸡 " * 12]) == "not English"
    assert prepare.verdict(["12 34 56 78 90 the 11 of 22 and 33 to 44 55 66 77 88 99 00."] * 5) == "not words"
    assert prepare.verdict(["The price of the #lamp is #low and the #chair is #cheap... and that is all we have to say with it..."] * 4) == "symbols"
    varied = [f"The fox counted hen number {i} and went on to the next one with care." for i in range(6)]
    assert prepare.verdict(varied + [varied[0]] * 4) == "repeats itself"


def test_scrub():
    text, n = prepare.scrub("Write to jane.doe+news@mail.example.org or to 192.168.0.12, not to version 1.2.3.")
    assert text == "Write to email@example.com or to 0.0.0.0, not to version 1.2.3."
    assert n == 2


def test_digest_ignores_case_and_spacing():
    assert prepare.digest("The  fox\nran.") == prepare.digest("the fox ran.")
    assert prepare.digest("The fox ran.") != prepare.digest("The fox sat.")


def test_minhash_estimates_the_overlap():
    words = [f"w{i}" for i in range(400)]
    a = " ".join(words)
    b = " ".join(words[:300] + [f"x{i}" for i in range(100)])  # three quarters the same
    mh = prepare.MinHash(n=512)
    sa, sb = set(prepare.shingles(a).tolist()), set(prepare.shingles(b).tolist())
    true = len(sa & sb) / len(sa | sb)
    assert abs(prepare.similarity(mh.signature(a), mh.signature(b)) - true) < 0.06
    assert prepare.similarity(mh.signature(a), mh.signature(a)) == 1.0
    assert prepare.similarity(mh.signature(a), mh.signature(" ".join(f"y{i}" for i in range(400)))) < 0.02


def test_near_duplicates_keeps_the_first_of_each():
    base = [f"The fox counted hen number {i} and went on to the next one with care." for i in range(60)]
    texts = [
        " ".join(base),
        " ".join(f"A page about boats, sentence {i}, with nothing in common with any fox." for i in range(60)),
        " ".join(base[:58] + ["A new last line that the first page did not have at all.", "And one more."]),  # a near copy of the first
        " ".join(base),  # an exact copy
    ]
    mh = prepare.MinHash()
    assert prepare.near_duplicates([mh.signature(t) for t in texts]) == {2, 3}


def test_repeated_lines():
    notice = "We use cookies to improve your experience on this site."
    texts = [f"Page {i} has its own sentence about subject {i}.\n{notice}" for i in range(4)]
    assert prepare.repeated_lines(texts) == {notice: 4}
