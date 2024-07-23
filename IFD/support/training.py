import tensorflow as tf
from IFD.losses.LossesAndMetrics import mean_abs_err_var



def compile_and_fit(model, train_ds, val_ds,loss_fn,prof_call_back,metric=mean_abs_err_var,patience=5,max_epochs=15): #prefetch(buffer_size=tf.data.AUTOTUNE)
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(),metrics=metric)

  history = model.fit(train_ds, epochs=max_epochs,
                      validation_data=val_ds,
                      callbacks=[early_stopping]) #,prof_call_back
  return history
