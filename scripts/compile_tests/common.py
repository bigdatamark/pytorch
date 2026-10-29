# [MODULE DOC] Helpers for analyzing JUnit XML test reports of Dynamo CI runs: load all XML
# files in a directory, collect testcase elements, classify them (skipped, passed, excluded by
# policy, unexpected success, passing-but-skipped, failure) and build a stable test key.
# Import-time side effects: defines parse using lxml if available (huge trees allowed),
# otherwise falls back to the standard library parser and emits a warning.
import functools
import os
import warnings


try:
    import lxml.etree

    p = lxml.etree.XMLParser(huge_tree=True)
    parse = functools.partial(lxml.etree.parse, parser=p)
except ImportError:
    import xml.etree.ElementTree as ET

    parse = ET.parse
    warnings.warn(
        "lxml was not found. `pip install lxml` to make this script run much faster"
    )


def open_test_results(directory):
    # [FUNCTION DOC] Recursively walks directory and parses every file ending in .xml.
    # Returns a list of parsed trees. Parse errors propagate.
    xmls = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                tree = parse(f"{root}/{file}")
                xmls.append(tree)
    return xmls


def get_testcases(xmls):
    # [FUNCTION DOC] Returns a flat list of all testcase elements found in the given parsed
    # report trees.
    testcases = []
    for xml in xmls:
        root = xml.getroot()
        testcases.extend(list(root.iter("testcase")))
    return testcases


def find(testcase, condition):
    # [FUNCTION DOC] HELPER: Applies the condition function to the list of all descendant
    # elements of testcase (excluding itself) and returns its result. Fails an assertion if
    # the first iterated element is not the testcase itself.
    children = list(testcase.iter())
    assert children[0] is testcase
    children = children[1:]
    return condition(children)


def skipped_test(testcase):
    # [FUNCTION DOC] True if the testcase has a skipped child element.
    def condition(children):
        # [FUNCTION DOC] HELPER: True if any child tag is skipped.
        return "skipped" in {child.tag for child in children}

    return find(testcase, condition)


def passed_test(testcase):
    # [FUNCTION DOC] True if the testcase has no children, or none of its children is tagged
    # skipped or failed.
    # KNOWN ISSUE: JUnit failures are normally tagged failure, not failed, so failing tests
    # may be counted as passed.
    def condition(children):
        # [FUNCTION DOC] HELPER: True for no children, else when neither skipped nor failed
        # tags are present.
        if len(children) == 0:
            return True
        tags = {child.tag for child in children}
        return "skipped" not in tags and "failed" not in tags

    return find(testcase, condition)


def key(testcase):
    # [FUNCTION DOC] Builds a unique string for a testcase: file attribute (UNKNOWN if absent),
    # class name and test name joined by double colons. Raises KeyError if classname or name
    # is missing.
    file = testcase.attrib.get("file", "UNKNOWN")
    classname = testcase.attrib["classname"]
    name = testcase.attrib["name"]
    return "::".join([file, classname, name])


def get_passed_testcases(xmls):
    # [FUNCTION DOC] Returns the testcases from the given trees for which passed_test is true.
    testcases = get_testcases(xmls)
    passed_testcases = [testcase for testcase in testcases if passed_test(testcase)]
    return passed_testcases


def get_excluded_testcases(xmls):
    # [FUNCTION DOC] Returns the testcases from the given trees that excluded_testcase marks
    # as excluded by policy.
    testcases = get_testcases(xmls)
    excluded_testcases = [t for t in testcases if excluded_testcase(t)]
    return excluded_testcases


def excluded_testcase(testcase):
    # [FUNCTION DOC] True if the testcase has a skipped child whose message contains the
    # policy phrase saying the test is not run. Raises KeyError if a skipped child has no
    # message attribute.
    def condition(children):
        # [FUNCTION DOC] HELPER: Scans skipped children for the policy skip message.
        for child in children:
            if child.tag == "skipped":
                if "Policy: we don't run" in child.attrib["message"]:
                    return True
        return False

    return find(testcase, condition)


def is_unexpected_success(testcase):
    # [FUNCTION DOC] True if the testcase has a failure child whose message (lowercased)
    # contains the text unexpected success.
    def condition(children):
        # [FUNCTION DOC] HELPER: Scans failure children for the unexpected success message.
        for child in children:
            if child.tag != "failure":
                continue
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            if is_unexpected_success:
                return True
        return False

    return find(testcase, condition)


MSG = "This test passed, maybe we can remove the skip from dynamo_test_failures.py"


def is_passing_skipped_test(testcase):
    # [FUNCTION DOC] True if the testcase has a skipped child whose message contains the MSG
    # constant, meaning a test that is skipped by a list but actually passed.
    def condition(children):
        # [FUNCTION DOC] HELPER: Scans skipped children for the passing-skipped message.
        for child in children:
            if child.tag != "skipped":
                continue
            has_passing_skipped_test_msg = MSG in child.attrib["message"]
            if has_passing_skipped_test_msg:
                return True
        return False

    return find(testcase, condition)


# NB: not an unexpected success
def is_failure(testcase):
    # [FUNCTION DOC] True if the testcase has a failure child that is not an unexpected
    # success.
    def condition(children):
        # [FUNCTION DOC] HELPER: Scans failure children for one whose message is not an
        # unexpected success.
        for child in children:
            if child.tag != "failure":
                continue
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            if not is_unexpected_success:
                return True
        return False

    return find(testcase, condition)
