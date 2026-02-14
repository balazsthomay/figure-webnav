"""Tests for perception module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from webnav.perception import (
    CODE_PATTERN,
    ElementInfo,
    PageState,
    _extract_codes,
    _extract_instruction,
    _extract_interactive,
    _extract_progress,
    _extract_step_number,
    _parse_yaml_to_nodes,
    _walk_tree,
    index_elements,
    snapshot,
)
from tests.conftest import (
    STEP1_A11Y_TREE,
    STEP1_ARIA_YAML,
    STEP2_A11Y_TREE,
    STEP2_ARIA_YAML,
    STEP3_A11Y_TREE_WITH_CODE,
    STEP3_ARIA_YAML_WITH_CODE,
    STEP5_A11Y_TREE,
    STEP5_ARIA_YAML,
    make_mock_page,
)


class TestWalkTree:
    def test_flattens_nested_tree(self):
        tree = {
            "role": "root",
            "name": "r",
            "children": [
                {"role": "child", "name": "c1"},
                {"role": "child", "name": "c2", "children": [
                    {"role": "grandchild", "name": "gc1"},
                ]},
            ],
        }
        nodes = _walk_tree(tree)
        assert len(nodes) == 4
        assert nodes[0]["depth"] == 0
        assert nodes[3]["depth"] == 2

    def test_handles_empty_tree(self):
        nodes = _walk_tree({"role": "root", "name": ""})
        assert len(nodes) == 1


class TestParseYamlToNodes:
    def test_parses_step1_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        roles = [n["role"] for n in nodes]
        assert "heading" in roles
        assert "button" in roles
        assert "textbox" in roles

    def test_extracts_button_names(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        button_names = [n["name"] for n in nodes if n["role"] == "button"]
        assert "Reveal Code" in button_names
        assert "Submit Code" in button_names

    def test_extracts_paragraph_text(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        paragraphs = [n for n in nodes if n["role"] == "paragraph"]
        assert any("click" in p["name"].lower() for p in paragraphs)

    def test_handles_empty_yaml(self):
        assert _parse_yaml_to_nodes("") == []

    def test_handles_colon_format(self):
        yaml = '- paragraph: Some instruction text here'
        nodes = _parse_yaml_to_nodes(yaml)
        assert len(nodes) == 1
        assert nodes[0]["role"] == "paragraph"
        assert nodes[0]["name"] == "Some instruction text here"


class TestExtractInstruction:
    def test_finds_click_instruction_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        instruction = _extract_instruction(nodes)
        assert "click" in instruction.lower()
        assert "reveal" in instruction.lower()

    def test_finds_scroll_instruction_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP2_ARIA_YAML)
        instruction = _extract_instruction(nodes)
        assert "scroll" in instruction.lower()
        assert "500" in instruction

    def test_finds_wait_instruction_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP3_ARIA_YAML_WITH_CODE)
        instruction = _extract_instruction(nodes)
        assert "wait" in instruction.lower()

    def test_finds_hidden_instruction_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP5_ARIA_YAML)
        instruction = _extract_instruction(nodes)
        assert "hidden" in instruction.lower()

    # Dict-based tests (legacy helpers still work)
    def test_finds_click_instruction(self):
        nodes = _walk_tree(STEP1_A11Y_TREE)
        instruction = _extract_instruction(nodes)
        assert "click" in instruction.lower()

    def test_skips_noise(self):
        nodes = _walk_tree(STEP1_A11Y_TREE)
        instruction = _extract_instruction(nodes)
        assert "lorem ipsum" not in instruction.lower()
        assert "content block" not in instruction.lower()


class TestExtractCodes:
    def test_finds_visible_code_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP3_ARIA_YAML_WITH_CODE)
        codes = _extract_codes(nodes)
        assert "AB3F9X" in codes

    def test_no_codes_when_hidden_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        codes = _extract_codes(nodes)
        assert len(codes) == 0

    def test_finds_visible_code(self):
        nodes = _walk_tree(STEP3_A11Y_TREE_WITH_CODE)
        codes = _extract_codes(nodes)
        assert "AB3F9X" in codes

    def test_no_codes_when_hidden(self):
        nodes = _walk_tree(STEP1_A11Y_TREE)
        codes = _extract_codes(nodes)
        assert len(codes) == 0

    def test_filters_false_positives(self):
        nodes = [{"name": "SUBMIT"}, {"name": "BUTTON"}, {"name": "XY45AB"}]
        codes = _extract_codes(nodes)
        assert "SUBMIT" not in codes
        assert "BUTTON" not in codes
        assert "XY45AB" in codes


class TestExtractInteractive:
    def test_finds_buttons_and_textbox_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        interactive = _extract_interactive(nodes)
        roles = {e["role"] for e in interactive}
        assert "button" in roles
        assert "textbox" in roles

    def test_filters_noise_buttons_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        interactive = _extract_interactive(nodes)
        names = {e["name"] for e in interactive}
        assert "Click Me!" not in names
        assert "Button!" not in names

    def test_deduplicates(self):
        nodes = [
            {"role": "button", "name": "Submit"},
            {"role": "button", "name": "Submit"},
        ]
        interactive = _extract_interactive(nodes)
        assert len(interactive) == 1


class TestExtractProgress:
    def test_finds_scroll_progress_from_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP2_ARIA_YAML)
        progress = _extract_progress(nodes)
        assert "Scrolled" in progress or "scrolled" in progress.lower()

    def test_empty_when_no_progress_yaml(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        progress = _extract_progress(nodes)
        assert progress == ""


class TestExtractStepNumber:
    def test_from_url(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        step = _extract_step_number(nodes, "https://example.com/step5?version=2")
        assert step == 5

    def test_from_content(self):
        nodes = _parse_yaml_to_nodes(STEP1_ARIA_YAML)
        step = _extract_step_number(nodes, "https://example.com/")
        assert step == 1  # From "Challenge Step 1"


class TestElementInfo:
    def test_creates_with_defaults(self):
        el = ElementInfo(index=0, tag="button")
        assert el.index == 0
        assert el.tag == "button"
        assert el.role == ""
        assert el.name == ""
        assert el.selector == ""
        assert el.visible is True
        assert el.bbox is None

    def test_creates_with_all_fields(self):
        el = ElementInfo(
            index=5,
            tag="input",
            role="textbox",
            name="Enter code",
            type="text",
            placeholder="6-char code",
            selector="#code-input",
            visible=True,
            bbox={"x": 10, "y": 20, "width": 200, "height": 30},
        )
        assert el.index == 5
        assert el.type == "text"
        assert el.placeholder == "6-char code"
        assert el.bbox["width"] == 200


class TestIndexElements:
    @pytest.mark.asyncio
    async def test_parses_valid_element_list(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=[
            {
                "index": 0,
                "tag": "button",
                "role": "button",
                "name": "Reveal Code",
                "type": "",
                "placeholder": "",
                "selector": "button:nth-of-type(1)",
                "visible": True,
                "bbox": {"x": 10, "y": 20, "width": 100, "height": 40},
            },
            {
                "index": 1,
                "tag": "input",
                "role": "",
                "name": "",
                "type": "text",
                "placeholder": "Enter code",
                "selector": "input:nth-of-type(1)",
                "visible": True,
                "bbox": {"x": 10, "y": 80, "width": 200, "height": 30},
            },
        ])
        elements = await index_elements(page)
        assert len(elements) == 2
        assert elements[0].tag == "button"
        assert elements[0].name == "Reveal Code"
        assert elements[1].tag == "input"
        assert elements[1].placeholder == "Enter code"

    @pytest.mark.asyncio
    async def test_handles_evaluate_exception(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(side_effect=Exception("JS error"))
        elements = await index_elements(page)
        assert elements == []

    @pytest.mark.asyncio
    async def test_handles_non_list_result(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        elements = await index_elements(page)
        assert elements == []

    @pytest.mark.asyncio
    async def test_handles_empty_list(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=[])
        elements = await index_elements(page)
        assert elements == []

    @pytest.mark.asyncio
    async def test_skips_non_dict_items(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=[
            {"index": 0, "tag": "button", "name": "OK", "selector": "button"},
            "invalid",
            42,
        ])
        elements = await index_elements(page)
        assert len(elements) == 1


class TestPageStatePrompt:
    def test_renders_compact_prompt(self):
        state = PageState(
            step=2,
            instruction="Scroll down 500px",
            progress="Scrolled: 0px / 500px",
            interactive=[{"role": "button", "name": "Submit"}],
            visible_codes=["ABC123"],
        )
        prompt = state.to_prompt()
        assert "STEP: 2/30" in prompt
        assert "Scroll down 500px" in prompt
        assert "ABC123" in prompt

    def test_includes_aria_yaml_when_present(self):
        state = PageState(step=1, aria_yaml="- button \"Test\"")
        prompt = state.to_prompt()
        assert "ARIA_TREE" in prompt

    def test_renders_indexed_elements(self):
        elements = [
            ElementInfo(index=0, tag="button", name="Reveal Code", selector="button:nth-of-type(1)"),
            ElementInfo(index=1, tag="input", type="text", placeholder="Enter code", selector="input:nth-of-type(1)"),
            ElementInfo(index=2, tag="canvas", role="img", selector="canvas"),
        ]
        state = PageState(step=1, instruction="Click the button", elements=elements)
        prompt = state.to_prompt()
        assert "ELEMENTS:" in prompt
        assert '[0] button "Reveal Code"' in prompt
        assert "[1] input type=text" in prompt
        assert 'placeholder="Enter code"' in prompt
        assert "[2] canvas role=img" in prompt

    def test_filters_noise_from_page_text(self):
        state = PageState(
            step=1,
            instruction="Click the button",
            raw_text="Click the button below\nLorem ipsum dolor sit amet\nContent Block Loaded\nYour code is: AB3F9X",
        )
        prompt = state.to_prompt()
        assert "Lorem ipsum" not in prompt
        assert "Content Block Loaded" not in prompt
        assert "AB3F9X" in prompt


class TestCodePattern:
    def test_matches_valid_codes(self):
        assert CODE_PATTERN.search("AB3F9X")
        assert CODE_PATTERN.search("123456")
        assert CODE_PATTERN.search("ABCDEF")

    def test_rejects_invalid(self):
        assert not CODE_PATTERN.search("abc123")  # lowercase
        assert not CODE_PATTERN.search("AB3F9")  # too short


class TestExtractInstructionFromText:
    def test_finds_instruction_in_raw_text(self):
        from webnav.perception import _extract_instruction_from_text
        text = "Some noise\nClick the button below to reveal the code\nMore noise"
        instruction = _extract_instruction_from_text(text)
        assert "click" in instruction.lower()

    def test_skips_noise(self):
        from webnav.perception import _extract_instruction_from_text
        text = "Lorem ipsum dolor sit amet\nContent Block Loaded"
        instruction = _extract_instruction_from_text(text)
        assert instruction == ""

    def test_empty_text(self):
        from webnav.perception import _extract_instruction_from_text
        assert _extract_instruction_from_text("") == ""


@pytest.mark.asyncio
async def test_snapshot_with_fallback_instruction():
    """Test that instruction is extracted from raw text when YAML has none."""
    page = make_mock_page(
        url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
        aria_yaml="- heading \"Challenge Step 1\"",
        inner_text="Click the button below to reveal the code\nReveal Code",
    )
    state = await snapshot(page)
    assert "click" in state.instruction.lower()


@pytest.mark.asyncio
async def test_snapshot_handles_aria_exception():
    """Test snapshot gracefully handles aria_snapshot failure."""
    page = make_mock_page(
        url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
        inner_text="Click the button to reveal code",
    )
    # Make aria_snapshot raise
    mock_loc = page.locator.return_value
    mock_loc.aria_snapshot = AsyncMock(side_effect=Exception("timeout"))
    state = await snapshot(page)
    # Should still work via raw text
    assert state.step == 1


@pytest.mark.asyncio
async def test_snapshot_integration():
    page = make_mock_page(
        url="https://serene-frangipane-7fd25b.netlify.app/step3?version=2",
        aria_yaml=STEP3_ARIA_YAML_WITH_CODE,
        inner_text="Your code is: AB3F9X\nSubmit Code",
    )
    state = await snapshot(page)
    assert state.step == 3
    assert "wait" in state.instruction.lower()
    assert "AB3F9X" in state.visible_codes
    # Elements list is populated (may be empty if evaluate returns None)
    assert isinstance(state.elements, list)
