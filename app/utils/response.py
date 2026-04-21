from fastapi.responses import JSONResponse


def jsend_success(data=None, status_code=200):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "success",
            "data": data,
        },
    )


def jsend_fail(data, status_code=400):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "fail",
            "data": data,
        },
    )


def jsend_error(message, status_code=500, data=None):
    content = {"status": "error", "message": message}
    if data:
        content["data"] = data
    return JSONResponse(status_code=status_code, content=content)
