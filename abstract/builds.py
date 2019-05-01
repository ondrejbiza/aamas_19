class DRN:

    PRE_CONV = ((16,), (7,), (1,), (1,))

    BLOCKS_CAP_16 = ((16, 16, 16, 16, 16, 16, 16, 16, 16, 16),
                     (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
                     (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
    POST_CONV_CAP_16 = ((16, 16, 16, 16), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

    BLOCKS_CAP_32 = ((16, 32, 32, 32, 32, 32, 32, 32, 32, 32),
              (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
              (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
    POST_CONV_CAP_32 = ((32, 32, 32, 32), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

    BLOCKS_CAP_64 = ((16, 32, 64, 64, 64, 64, 64, 64, 64, 64),
              (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
              (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
    POST_CONV_CAP_64 = ((64, 64, 64, 64), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

    BLOCKS_CAP_128 = ((16, 32, 64, 64, 128, 128, 128, 128, 128, 128),
              (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
              (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
    POST_CONV_CAP_128 = ((128, 128, 128, 128), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

    BLOCKS_CAP_256 = ((16, 32, 64, 64, 128, 128, 256, 256, 256, 256),
                      (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
                      (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
    POST_CONV_CAP_256 = ((256, 256, 256, 256), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

    BLOCKS_CAP_512 = ((16, 32, 64, 64, 128, 128, 256, 256, 512, 512),
                      (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
                      (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
    POST_CONV_CAP_512 = ((512, 512, 512, 512), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

    @classmethod
    def get_build(cls, cap):

        if cap == 16:
            return cls.BLOCKS_CAP_16, cls.POST_CONV_CAP_16
        elif cap == 32:
            return cls.BLOCKS_CAP_32, cls.POST_CONV_CAP_16
        elif cap == 64:
            return cls.BLOCKS_CAP_64, cls.POST_CONV_CAP_16
        elif cap == 128:
            return cls.BLOCKS_CAP_128, cls.POST_CONV_CAP_16
        elif cap == 256:
            return cls.BLOCKS_CAP_256, cls.POST_CONV_CAP_16
        elif cap == 512:
            return cls.BLOCKS_CAP_512, cls.POST_CONV_CAP_16
        else:
            raise ValueError("Wrong filters cap.")
