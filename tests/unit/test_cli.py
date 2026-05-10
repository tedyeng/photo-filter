import pytest

from photofilter.cli import build_parser

def test_cli_help(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--help'])
    # argparse exits with code 0 on help
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert 'best‑composed' in captured.out
