using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class ImageSelection : MonoBehaviour, IPointerClickHandler
{
    [SerializeField]private Image SelectedImage;

     // This method will be called when the player clicks on the Image.
    public void OnPointerClick(PointerEventData eventData)
    {
        Image image = GetComponent<Image>();
        if (image != null)
        {
            SelectedImage.sprite = image.sprite;
        }
    }
}
