import zstandard as zstd

class ZstdBackend:
    def __init__(self, level=15):
        self.cctx = zstd.ZstdCompressor(level=level)
        self.dctx = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        return self.cctx.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self.dctx.decompress(data)