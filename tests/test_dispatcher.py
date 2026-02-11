"""Tests for dispatcher module (Tier 0 regex matching)."""

from __future__ import annotations

from webnav.dispatcher import Action, match


class TestClickPatterns:
    def test_click_to_reveal(self):
        actions = match("Click the button below to reveal the code")
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].type == "click"
        assert "Reveal" in actions[0].selector

    def test_click_button_reveal(self):
        actions = match("Click the Reveal button to see your code")
        assert actions is not None
        assert actions[0].type == "click"

    def test_click_n_times(self):
        actions = match("Click the button 5 times to reveal the code")
        assert actions is not None
        assert actions[0].type == "click"
        assert actions[0].amount == 5

    def test_click_button_generic(self):
        actions = match("Click the button to proceed")
        assert actions is not None
        assert actions[0].type == "click"


class TestScrollPatterns:
    def test_scroll_with_pixels(self):
        actions = match("Scroll down at least 500px to reveal the code")
        assert actions is not None
        assert actions[0].type == "scroll"
        assert actions[0].amount >= 500

    def test_scroll_pixels_variant(self):
        actions = match("Scroll down 300 pixels")
        assert actions is not None
        assert actions[0].type == "scroll"
        assert actions[0].amount >= 300

    def test_scroll_down_generic(self):
        actions = match("Scroll down to find the code")
        assert actions is not None
        assert actions[0].type == "scroll"
        assert actions[0].amount > 0


class TestWaitPattern:
    def test_wait_seconds(self):
        actions = match("Wait 5 seconds for the code to appear")
        assert actions is not None
        assert actions[0].type == "wait"
        assert actions[0].amount >= 5  # includes buffer

    def test_wait_3_seconds(self):
        actions = match("Please wait 3 seconds")
        assert actions is not None
        assert actions[0].amount >= 3  # includes buffer


class TestHiddenPattern:
    def test_hidden_in_dom(self):
        actions = match("The code is hidden in the DOM")
        assert actions is not None
        assert actions[0].type == "js"

    def test_inspect_dom(self):
        actions = match("Inspect the DOM to find the hidden code")
        assert actions is not None
        assert actions[0].type == "js"

    def test_hidden_dom_before_click(self):
        """Hidden DOM instruction mentioning 'click to reveal' should match hidden, not click."""
        actions = match("Hidden DOM Challenge: The code is hidden somewhere. Inspect the DOM or click to reveal")
        assert actions is not None
        assert actions[0].type == "js"

    def test_hidden_element(self):
        actions = match("The code is hidden in this element somewhere")
        assert actions is not None
        assert actions[0].type == "js"


class TestHoverPattern:
    def test_hover_instruction(self):
        actions = match("Hover over the element to reveal the code")
        assert actions is not None
        assert actions[0].type == "js"  # JS sustained hover first
        assert any(a.type == "hover" for a in actions)  # Playwright backup

    def test_hover_with_seconds(self):
        actions = match("Hover over the box for at least 1 second to reveal the code")
        assert actions is not None
        assert actions[0].type == "js"  # JS sustained hover
        assert any(a.type == "hover" for a in actions)
        assert any(a.type == "wait" for a in actions)


class TestKeySequencePattern:
    def test_keyboard_sequence(self):
        actions = match("Keyboard Sequence Challenge: Press 4 keys in sequence to reveal the code.")
        assert actions is not None
        assert actions[0].type == "key_sequence"

    def test_press_keys_in_sequence(self):
        actions = match("Press 3 keys in sequence")
        assert actions is not None
        assert actions[0].type == "key_sequence"


class TestPressPattern:
    def test_press_key(self):
        actions = match("Press 'Enter' to continue")
        assert actions is not None
        assert actions[0].type == "press"
        assert actions[0].value == "Enter"


class TestNoMatch:
    def test_empty_instruction(self):
        assert match("") is None

    def test_unknown_instruction(self):
        assert match("Do something completely novel") is None

    def test_drag_uses_playwright(self):
        """Drag challenges should use Playwright mouse simulation."""
        actions = match("Fill all 6 slots with any pieces to reveal the code")
        assert actions is not None
        assert actions[0].type == "drag_fill"

    def test_drag_explicit(self):
        """Explicit drag instruction should use Playwright drag."""
        actions = match("Drag the pieces into the slots")
        assert actions is not None
        assert actions[0].type == "drag_fill"


class TestCanvasPattern:
    def test_canvas_draw_with_click(self):
        actions = match("Canvas Challenge: Draw at least 3 strokes on the canvas below to reveal the code.")
        assert actions is not None
        assert actions[0].type == "canvas_draw"
        assert actions[1].type == "click"  # Must click Reveal after drawing

    def test_draw_strokes(self):
        actions = match("Draw at least 3 strokes on the canvas")
        assert actions is not None
        assert actions[0].type == "canvas_draw"


class TestTimingPattern:
    def test_timing_challenge(self):
        actions = match('Timing Challenge: Click "Capture" while the window is active to reveal the real code!')
        assert actions is not None
        assert actions[0].type == "wait"
        assert any(a.type == "click" for a in actions)


class TestAudioPattern:
    def test_audio_challenge(self):
        actions = match('Audio Challenge: Play the audio, listen to the hint, then click "Complete" to reveal the real code.')
        assert actions is not None
        assert actions[0].type == "click"  # Play Audio


class TestVideoPattern:
    def test_video_challenge(self):
        actions = match("Video Challenge: Seek through the video frames. Find and navigate to frame 30, then complete the challenge.")
        assert actions is not None
        assert any(a.type == "click" for a in actions)


class TestMultiTabPattern:
    def test_multi_tab(self):
        actions = match("Multi-Tab Challenge: Visit all 3 tabs by clicking each button.")
        assert actions is not None
        assert actions[0].type == "js"


class TestGesturePattern:
    def test_gesture_draw_circle(self):
        actions = match("Gesture Challenge: Draw a circle on the canvas below, then click complete.")
        assert actions is not None
        assert actions[0].type == "canvas_draw"

    def test_gesture_draw_square_matches_gesture_not_canvas(self):
        """Gesture with 'canvas' in text must match gesture pattern, not canvas."""
        actions = match("Gesture Challenge: Draw a square on the canvas below, then click complete.")
        assert actions is not None
        assert actions[0].type == "canvas_draw"
        assert actions[1].type == "click"
        assert "Complete" in actions[1].selector


class TestSequencePattern:
    def test_sequence_challenge(self):
        actions = match("Sequence Challenge: Complete all 4 actions to reveal the code.")
        assert actions is not None
        assert actions[0].type == "js"  # JS does all 4 actions first
        assert any(a.type == "click" for a in actions)  # Backup click
        assert any(a.type == "hover" for a in actions)  # Backup hover
        assert len(actions) >= 6  # js, click, hover, wait, fill, scroll, js, click

    def test_keyboard_sequence_not_matched(self):
        """Keyboard Sequence Challenge must NOT match the sequence pattern."""
        actions = match("Keyboard Sequence Challenge: Press 4 keys in sequence to reveal the code.")
        assert actions is not None
        assert actions[0].type == "key_sequence"


class TestShadowDomPattern:
    def test_shadow_dom(self):
        actions = match("Shadow DOM Challenge: Navigate through 3 nested layers to reveal the code.")
        assert actions is not None
        assert actions[0].type == "js"


class TestWebSocketPattern:
    def test_websocket(self):
        actions = match("WebSocket Challenge: Connect to the simulated WebSocket server and receive the code.")
        assert actions is not None
        assert actions[0].type == "click"  # Connect button


class TestServiceWorkerPattern:
    def test_service_worker(self):
        actions = match("Service Worker Challenge: Register a service worker, wait for the cache to be populated, then retrieve the code.")
        assert actions is not None
        assert actions[0].type == "click"  # Register button


class TestMutationPattern:
    def test_mutation(self):
        actions = match("Mutation Challenge: Trigger 5 DOM mutations to reveal the code.")
        assert actions is not None
        assert actions[0].type == "click"
        assert actions[0].amount == 6  # Extra for safety


class TestRecursiveIframePattern:
    def test_recursive_iframe(self):
        actions = match("Recursive Iframe Challenge: Navigate through 3 nested levels to find the code at the deepest level.")
        assert actions is not None
        assert actions[0].type == "js"


class TestPuzzlePattern:
    def test_puzzle(self):
        actions = match("Puzzle Challenge: Solve this puzzle to reveal the code")
        assert actions is not None
        assert actions[0].type == "wait"  # Wait for React to render puzzle UI
        assert actions[1].type == "js"  # Puzzle solver JS


class TestSplitPartsPattern:
    def test_split_parts(self):
        actions = match("Split Parts Challenge: Find and click all 3 parts scattered on this page.")
        assert actions is not None
        assert actions[0].type == "js"


class TestEncodedPattern:
    def test_base64(self):
        actions = match("Encoded Code Challenge: Decode this Base64 string to find a hint, then enter a 6-character code:")
        assert actions is not None
        assert actions[0].type == "js"


class TestRotatingPattern:
    def test_rotating(self):
        actions = match('Rotating Code Challenge: The code changes every 3 seconds. Click "Capture" at least 3 times to complete the challenge.')
        assert actions is not None
        assert actions[0].type == "click"
        assert actions[0].amount == 3


class TestObfuscatedPattern:
    def test_obfuscated(self):
        actions = match("Obfuscated Code Challenge: The code has been obfuscated. Decode it and enter your answer.")
        assert actions is not None
        assert actions[0].type == "js"


class TestActionDataclass:
    def test_to_dict_strips_empty(self):
        action = Action(type="click", selector="button")
        d = action.to_dict()
        assert "value" not in d
        assert "amount" not in d
        assert d["type"] == "click"
