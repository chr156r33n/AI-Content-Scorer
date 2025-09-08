import re
from collections import Counter
from nlp_highlight import annotate_passage, render_html


TEST_PASSAGE = (
    "Our research team, which is occasionally and somewhat loosely coordinated by a committee that has been formed by several departments and is often consulted by external partners, might usually suggest that the new catalyst is likely improving reaction rates, although it could appear to be roughly kind of approximately twice as efficient under typical lab conditions. "
    "The apparatus, being adjusted by a technician who was being supervised by a manager and who, in turn, had been advised by a vendor while the calibration was still ongoing and, perhaps, delayed because of procurement, was tested with seven samples and, as it seems, produced results that are somewhat inconsistent, which may suggest, although we cannot be entirely certain, that the temperature control system, in its current configuration, tends to drift outside the nominal range when the ventilation is partially obstructed and when additional peripherals are connected, and this, in many cases, generally causes unexpected variance that is, in effect, difficult to mitigate without, in practice, redesigning the enclosure. "
    "Meanwhile, I went to the grocery store to buy bananas and discussed movie trailers, which were pretty cool, with a friend. "
    "In conclusion, the updated documentation, which had been reviewed by multiple stakeholders and is usually considered comprehensive, appears to align with the broader objectives, but it might require minor revisions."
)


def main() -> None:
    spans = annotate_passage(TEST_PASSAGE)
    counts = Counter(s["label"] for s in spans)

    # Compute overlaps among spans with different labels
    overlap_pairs = 0
    for i in range(len(spans)):
        a = spans[i]
        for j in range(i + 1, len(spans)):
            b = spans[j]
            if a["start"] < b["end"] and a["end"] > b["start"]:
                if a["label"] != b["label"]:
                    overlap_pairs += 1

    html = render_html(TEST_PASSAGE, spans)
    multi_class_segments = len(re.findall(r'data-roles="[^"]*,[^"]*"', html))
    # Strip tags to verify no duplication or omissions
    stripped = re.sub(r"</span>", "", html)
    stripped = re.sub(r"<span[^>]*>", "", stripped)
    matches_original = (stripped == TEST_PASSAGE)

    print("Total spans:", len(spans))
    print("Counts:")
    for label in ["Subject", "Predicate", "Object", "Hedging", "TopicDrift", "TooLong", "TooComplex"]:
        print(f"  {label}: {counts.get(label, 0)}")
    print("Overlapping span pairs (different labels):", overlap_pairs)
    print("Segments with multiple roles in HTML:", multi_class_segments)
    print("Stripped HTML equals original:", matches_original)
    if not matches_original:
        max_len = min(len(stripped), len(TEST_PASSAGE))
        idx = 0
        while idx < max_len and stripped[idx] == TEST_PASSAGE[idx]:
            idx += 1
        print("First diff at index:", idx)
        print("Around diff (rendered):", stripped[max(0, idx-40): idx+40].replace("\n", "\\n"))
        print("Around diff (original):", TEST_PASSAGE[max(0, idx-40): idx+40].replace("\n", "\\n"))
    print("HTML preview:")
    print(html[:600])


if __name__ == "__main__":
    main()

