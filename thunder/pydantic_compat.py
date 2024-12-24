def resolve_pydantic_major() -> int:
    import pydantic
    return int(pydantic.__version__.split(".")[0])


PYDANTIC_MAJOR = resolve_pydantic_major()

if PYDANTIC_MAJOR == 2:
    from pydantic import BaseModel, field_validator as _field_validator

    def field_validator(*args, always=None, **kwargs):
        # we just ignore `always`
        return _field_validator(*args, **kwargs)

    def model_validate(cls, data):
        return cls.model_validate(data)

    def model_dump(obj, **kwargs):
        return obj.model_dump(**kwargs)

    def model_copy(cls, **kwargs):
        return cls.model_copy(**kwargs)

    class NoExtra(BaseModel):
        model_config = {
            'extra': 'forbid'
        }

elif PYDANTIC_MAJOR == 1:
    from pydantic import BaseModel, root_validator, validator as _field_validator

    # we don't use this with pydantic==1 anyway
    core_schema = None

    def model_validator(mode: str):
        assert mode == 'before'
        return root_validator(pre=True)

    def field_validator(*args, mode: str = 'after', **kwargs):
        # we just ignore `always`
        assert mode in ('before', 'after')
        if mode == 'before':
            kwargs['pre'] = True
        return _field_validator(*args, **kwargs)

    def model_validate(cls, data):
        return cls.parse_obj(data)

    def model_dump(obj, **kwargs):
        return obj.dict(**kwargs)

    def model_copy(cls, **kwargs):
        return cls.copy(**kwargs)

    class NoExtra(BaseModel):
        class Config:
            extra = 'forbid'
else:
    import pydantic
    raise RuntimeError(f"Expected pydantic<3.0.0, got {pydantic.__version__}")
