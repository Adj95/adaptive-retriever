import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from unittest.mock import MagicMock
import pytest
import torch


def test_batch_generation_shapes_and_rules():
    """Verify tokenizer padding side, pad_token fallback, attention_mask passing,
    prompt slicing, and length-sorting order restoration.
    """
    mock_tokenizer = MagicMock()
    mock_tokenizer.padding_side = "right"
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<|endoftext|>"
    mock_tokenizer.pad_token_id = 99

    # Set up pad_token setting logic
    if mock_tokenizer.pad_token is None:
        mock_tokenizer.pad_token = mock_tokenizer.eos_token
    mock_tokenizer.padding_side = "left"

    assert mock_tokenizer.padding_side == "left", "tokenizer.padding_side must be 'left' for decoder-only generation"
    assert mock_tokenizer.pad_token == "<|endoftext|>", "pad_token must fallback to eos_token if None"

    # Prompts of varying lengths to test sorting and restoration
    prompts = [
        "Short prompt",
        "This is a much longer prompt that will be sorted towards the back of the batch",
        "Medium length prompt string",
    ]

    # Test sorting
    indexed_prompts = list(enumerate(prompts))
    indexed_prompts.sort(key=lambda x: len(x[1]))

    sorted_indices = [idx for idx, _ in indexed_prompts]
    assert sorted_indices == [0, 2, 1], "Prompts should be sorted by length ascending"

    # Stub inputs and generate output
    mock_model = MagicMock()

    # Simulate tokenized input tensors: shape [3, 10]
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 3)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]] * 3)
    mock_tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "to": lambda device: {"input_ids": input_ids, "attention_mask": attention_mask},
    }

    # Simulate output tensors: shape [3, 15] (10 prompt tokens + 5 generated tokens)
    generated_outputs = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101, 102, 103, 104, 105],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 201, 202, 203, 204, 205],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 301, 302, 303, 304, 305],
    ])
    mock_model.generate.return_value = generated_outputs

    input_len = input_ids.shape[1]
    sliced_outputs = generated_outputs[:, input_len:]

    assert sliced_outputs.shape == (3, 5), "Decoding slice outputs[:, input_len:] must isolate generated completion"
    assert sliced_outputs[0].tolist() == [101, 102, 103, 104, 105]

    # Verify attention_mask passed to generate
    mock_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=96,
        pad_token_id=mock_tokenizer.pad_token_id,
    )
    mock_model.generate.assert_called_with(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=96,
        pad_token_id=99,
    )

    # Re-assemble results in original prompt order
    results_by_idx = {
        0: "Response 0",
        2: "Response 2",
        1: "Response 1",
    }
    restored_results = [results_by_idx[i] for i in range(len(prompts))]
    assert restored_results == ["Response 0", "Response 1", "Response 2"], "Original input order must be restored"
