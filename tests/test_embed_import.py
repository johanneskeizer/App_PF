def test_embedder_loads():
    from embeddings import embed_with
    assert callable(embed_with)
