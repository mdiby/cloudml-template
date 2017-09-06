import tensorflow as tf
import metadata
import model
import input

def get_eval_metrics():

    if metadata.TASK_TYPE == "regression":

        metrics = {
            'rmse': tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_root_mean_squared_error)
        }

    else:

        # metrics = {
        #     'auc': tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_auc,
        #                                        prediction_key="classes",
        #                                        label_key=None
        #                                        )
        # }

        metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy,
                                               prediction_key="classes", # probabilities | classes | logit
                                               label_key=None
                                               )
        }

    return metrics


def generate_experiment_fn(**experiment_args):

    def _experiment_fn(run_config, hparams):

        train_input = lambda: input.generate_text_input_fn(
            hparams.train_files,
            mode = tf.contrib.learn.ModeKeys.TRAIN,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.train_batch_size
        )

        eval_input = lambda: input.generate_text_input_fn(
            hparams.eval_files,
            mode=tf.contrib.learn.ModeKeys.EVAL,
            batch_size=hparams.eval_batch_size
        )

        if metadata.TASK_TYPE == "classification":
            estimator = model.create_classifier(
                config=run_config
            )
        else:
            estimator = model.create_regressor(
                config=run_config
            )

        return tf.contrib.learn.Experiment(
            estimator,
            train_input_fn=train_input,
            eval_input_fn=eval_input,
            eval_metrics=get_eval_metrics(),
            **experiment_args
        )

    return _experiment_fn
