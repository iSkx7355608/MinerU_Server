import copy
import json
import os
from tempfile import NamedTemporaryFile
import uuid

import magic_pdf.model as model_config
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from loguru import logger
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from concurrent.futures import ThreadPoolExecutor  # 导入ThreadPoolExecutor


model_config.__use_inside_model__ = True

app = FastAPI()

#创建队列
import asyncio  # 导入asyncio模块
from collections import defaultdict


# 创建线程池
executor = ThreadPoolExecutor(max_workers=5)  # 可以根据需要调整max_workers的数量


# 创建异步队列
queue = asyncio.Queue()  # 使用asyncio.Queue
task_status = defaultdict(lambda: {"status": "pending", "result": None, "error": None})

def json_md_dump(
        pipe,
        md_writer,
        pdf_name,
        content_list,
        md_content,
):
    # Write model results to model.json
    orig_model_list = copy.deepcopy(pipe.model_list)
    md_writer.write(
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json"
    )

    # Write intermediate results to middle.json
    md_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json"
    )

    # Write text content results to content_list.json
    md_writer.write(
        content=json.dumps(content_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_content_list.json"
    )

    # Write results to .md file
    md_writer.write(
        content=md_content,
        path=f"{pdf_name}.md"
    )

def pdf_parse_main(
        task_id: str = None,
        pdf_name: str = None,
        temp_pdf_path: str = None,
        parse_method: str = 'auto',
        model_json_path: str = None,
        is_json_md_dump: bool = True,
        output_dir: str = "output"
):
    """
    Execute the process of converting PDF to JSON and MD, outputting MD and JSON files to the specified directory
    :param pdf_file: The PDF file to be parsed
    :param parse_method: Parsing method, can be auto, ocr, or txt. Default is auto. If results are not satisfactory, try ocr
    :param model_json_path: Path to existing model data file. If empty, use built-in model. PDF and model_json must correspond
    :param is_json_md_dump: Whether to transport parsed data to forend
    :param output_dir: Output directory for results. A folder named after the PDF file will be created to store all results
    """
    try:
        # Create a temporary file to store the uploaded PDF
        

        if output_dir:
            output_path = os.path.join(output_dir, task_id, pdf_name)
        else:
            output_path = os.path.join(os.path.dirname(temp_pdf_path), task_id, pdf_name)

        output_image_path = os.path.join(output_path, 'images')

        # Get parent path of images for relative path in .md and content_list.json
        image_path_parent = os.path.basename(output_image_path)

        pdf_bytes = open(temp_pdf_path, "rb").read()  # Read binary data of PDF file

        if model_json_path:
            # Read original JSON data of PDF file parsed by model, list type
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # Execute parsing steps
        image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(output_path)

        # Choose parsing method
        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            logger.error("Unknown parse method, only auto, ocr, txt allowed")
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["error"] = "Unknown parse method, only auto, ocr, txt allowed"
            return

        # Execute classification
        pipe.pipe_classify()

        # If no model data is provided, use built-in model for parsing
        if not model_json:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()  # Parse
            else:
                logger.error("Need model list input")
                task_status[task_id]["status"] = "failed"
                task_status[task_id]["error"] = "Model list input required"
                return

        # Execute parsing
        pipe.pipe_parse()

        # Save results in text and md format
        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)
        data = {"layout": copy.deepcopy(pipe.model_list), "info": pipe.pdf_mid_data, "content_list": content_list,'md_content':md_content}
        logger.info(f"{pdf_name} parsed")
        # logger.info(f"{data}")
        task_status[task_id]["status"] = "completed"
        if is_json_md_dump:
            task_status[task_id]["result"] = data
        return 

    except Exception as e:
        logger.exception(e)
        task_status[task_id]["status"] = "failed"
        task_status[task_id]["error"] = str(e)
        return
    finally:
        # Clean up the temporary file
        if 'temp_pdf_path' in locals():
            os.unlink(temp_pdf_path)




# 异步处理函数
@app.post("/async_pdf_parse", tags=["projects"], summary="异步接收文件解析请求，并为每一个任务返回一个task_id")
async def async_pdf_parse(
        background_tasks: BackgroundTasks,
        pdf_file: UploadFile = File(...),
        parse_method: str = 'auto',
        model_json_path: str = None,
        is_json_md_dump: bool = False,
        output_dir: str = "output"
):
    task_id = uuid.uuid4()  # 生成唯一任务ID
    
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(await pdf_file.read())
            temp_pdf_path = temp_pdf.name

    pdf_name = os.path.basename(pdf_file.filename).split(".")[0]
    
    task_data = {
        "task_id": str(task_id),
        "temp_pdf_path":temp_pdf_path,
        "pdf_name": pdf_name,
        "parse_method": parse_method,
        "model_json_path": model_json_path,
        "is_json_md_dump": is_json_md_dump,
        "output_dir": output_dir
    }
    
    
    await queue.put(task_data)  # 将任务数据放入队列
    
    # 启动后台工作线程（如果未启动）
    if not hasattr(app, "worker_started"):
        app.worker_started = True
        print("starting worker")
        background_tasks.add_task(task_consumer)
         
    return JSONResponse({"task_id": str(task_id)}, status_code=200)

# 异步任务消费者
async def task_consumer():
    while True:
        print("task_consumer")
        task_data = await queue.get()  # 从队列中获取任务
        if task_data:
            logger.info(f"Processing task: {task_data}")
            task_status[task_data['task_id']]["status"] = "processing"
            # 使用线程池执行解析任务
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, pdf_parse_main, 
                                       task_data['task_id'],
                task_data['pdf_name'],
                task_data['temp_pdf_path'],
                task_data['parse_method'],
                task_data['model_json_path'],
                task_data['is_json_md_dump'],
                task_data['output_dir'])
            queue.task_done()  # 标记任务完成
  
@app.get("/task_status/{task_id}", tags=["projects"], summary="查询任务状态")
async def get_task_status(task_id: str):
    
        if task_id not in task_status:
            return JSONResponse(content={"error": "Task not found"}, status_code=200)
        
        
        data = copy.deepcopy(task_status[task_id])
        if task_status[task_id]['status']=='completed':
            data = copy.deepcopy(task_status[task_id])
            task_status.pop(task_id)
            print(task_status)
            return data
        return task_status[task_id]

# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8888)