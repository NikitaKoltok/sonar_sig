import tensorflow as tf
import numpy as np
import traceback
import torch
import os



class Logger(object):
    """
    Общее описание класса
    """
    def __init__(self, log_dir, save_weight):
        """
        Create a summary writer logging to log_dir
        :param log_dir: str: папка для местоположения логов
        :param save_weight: str: полное имя для сохранения весов модели
        """
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        self.writer = tf.summary.FileWriter(log_dir)

        self.root = save_weight
        self.best_acc = 0

    def save_checkpoint(self, val_acc, model, optimizer, cur_epoch, cur_snap, cur_fold):
        """
        Сохранить лучшее состояние сети по точности на валидации
        :param val_acc: int: значение точности на валидации
        :param model: модель сети
        :param optimizer: текущее состояние optimizer
        :param cur_epoch: int: текущая эпоха обучения
        :return: bool: True - все прошло успешно, False - в противном случае
        """
        try:
            if val_acc > self.best_acc:
                print('Saving model with validation accuracy:', val_acc)
                torch.save({
                    'epoch': cur_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(self.save_weight, f's{cur_snap}_' + f'f{cur_fold}_' + 'checkpoint.pth'))
                self.best_acc = val_acc

            return True

        except Exception as err:
            print('Ошибка:\n', traceback.format_exc())
            return False

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def hist_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
