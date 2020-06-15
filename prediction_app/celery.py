from celery import Celery
from celery.schedules import crontab
from prediction_app import downloadFiles

#todo get redis, maybe start using docker containers instead of venv?
celery_worker = Celery('predict_stocks', broker=redis)

@celery_worker.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(crontab(hour=16, minute=30, day_of_week="1-5"), start_predictions.s())

@celery_worker.task
def start_predictions():
    downloadFiles.main()
