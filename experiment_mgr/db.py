import peewee as pw
import json

db = pw.SqliteDatabase("experiments.sqlite", pragmas={'foreign_keys': 1})

class ExperimentConfig(pw.Model):
    id = pw.PrimaryKeyField(null=False)

    model_config = pw.CharField(null=False)
    data_config = pw.CharField(null=False)
    trained_checkpoint = pw.CharField(null=False)
    pad_to_shape = pw.CharField(null=False)
    processor_type = pw.CharField(null=False)
    annot_type = pw.CharField(null=False)

    class Meta:
        database = db
        indexes = (
            (("model_config",
              "data_config",
              "trained_checkpoint",
              "pad_to_shape",
              "processor_type",
              "annot_type"), True),
        )

class ArgsGroup(pw.Model):
    id = pw.PrimaryKeyField(null=False)

    class Meta:
        database = db

class KeyWordArgs(pw.Model):
    id = pw.PrimaryKeyField(null=False)
    group = pw.ForeignKeyField(ArgsGroup, backref='kwargs')

    name = pw.CharField(null=False)
    value = pw.CharField(null=False)
    value_type = pw.CharField(null=False)

    class Meta:
        database = db
        indexes = (
            (("group",
              "name"), True),
        )

class Experiment(pw.Model):
    id = pw.PrimaryKeyField(null=False)
    config = pw.ForeignKeyField(ExperimentConfig, backref='experiments')
    arg_group = pw.ForeignKeyField(ArgsGroup, backref='experiments')

    class Meta:
        database = db
        indexes = (
            (("config",
              "arg_group"), True),
        )

class Result(pw.Model):
    experiment = pw.ForeignKeyField(Experiment, backref='result', primary_key=True)

    auroc = pw.DoubleField(null=False)
    aupr = pw.DoubleField(null=False)
    fpr_at_tpr = pw.DoubleField(null=False)
    detection_error = pw.DoubleField(null=False)
    max_iou = pw.DoubleField(null=False)
    had_error = pw.BooleanField(null=False)

    class Meta:
        database = db

#keep long text separate so it is not pulled with result
class PrintBuffer(pw.Model):
    result = pw.ForeignKeyField(Result, backref='print_buffer', primary_key=True)
    value = pw.TextField()

    class Meta:
        database = db

db.connect()
db.create_tables([ExperimentConfig, ArgsGroup, KeyWordArgs, Experiment, Result, PrintBuffer])
