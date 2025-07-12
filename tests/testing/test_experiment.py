

def test_hello():
    """Test the hello function."""
    from tcl.testing import example
    assert example.hello() == "Hello, World!"
